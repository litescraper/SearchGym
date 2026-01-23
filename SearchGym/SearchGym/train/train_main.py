import asyncio
import os
import sys
import uuid
import json
import gc
import torch
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from dataclasses import dataclass, field
from typing import List

import hashlib

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ModelRequest,
    WeightUpdateMeta,
    StepInfo,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig
from areal.platforms import current_platform
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.data import concat_padded_tensors
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils import seeding, logging, stats_tracker

from SearchGym.train.prompts import SEARCH_ACCESS_PROMPT_TEMPLATE, SEARCH_ONLY_PROMPT_TEMPLATE, INVALID_PROMPT, VALID_PROMPT, NORMAL_ACCESS, STRICT_ACCESS
from SearchGym.train.search_agent import SearchAgent
from SearchGym.utils.search_tool import SearchToolBox
from SearchGym.utils.rewards import correct_format_fn

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"SearchGym @ {worker_id}")

def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class SearchGymWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 128,
        n_trajs: int = 1,
        search_client_type: str = "async-online-search-access",
        reward_type: str = "F1",
        topk: int = 5,
        valid_inst_ratio: float = 1.0,
        max_tokens: int = 32000,
        search_only: bool = False,
        steps_per_epoch: int | None = None,
        use_queue: bool = False,
        redis_config: dict = None,
        access_strict_start_ratio: float = 0.9,
        access_strict_end_ratio: float = 0.1,
        access_strict_decay_steps: int = 100,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.use_queue = use_queue
        self.redis_config = redis_config or {}
        self.max_tokens = max_tokens
        self.search_only = search_only
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Search hyper-parameters
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.reward_type = reward_type
        self.topk = topk
        self.valid_inst_ratio = valid_inst_ratio
        self.search_client_type = search_client_type
        self.steps_per_epoch = steps_per_epoch

        # Access probability decay parameters
        self.access_strict_start_ratio = access_strict_start_ratio
        self.access_strict_end_ratio = access_strict_end_ratio
        self.access_strict_decay_steps = access_strict_decay_steps

        # 打印初始化参数
        logger.info(f"[ACCESS DECAY] Initialized: start_ratio={self.access_strict_start_ratio}, "
                   f"end_ratio={self.access_strict_end_ratio}, decay_steps={self.access_strict_decay_steps}")

        self.toolbox = SearchToolBox(dataset_path=dataset_path, reward_type=self.reward_type, topk=self.topk, search_client_type=self.search_client_type,use_queue=self.use_queue, redis_config=self.redis_config, search_only=self.search_only)

    def get_access_strict_ratio(self, global_step):
        """Calculate the current access strict ratio based on global step"""
        if global_step >= self.access_strict_decay_steps:
            return self.access_strict_end_ratio

        # Linear decay from start_ratio to end_ratio
        progress = global_step / self.access_strict_decay_steps
        current_ratio = self.access_strict_start_ratio - progress * (self.access_strict_start_ratio - self.access_strict_end_ratio)
        return current_ratio
    async def collect_agent_trajectory(self, valid_inst, qid, prompt, prompt_token_ids, engine):
        agent = SearchAgent(prompt, prompt_token_ids)
        score = 0
        ground_truth = None
        # a unique trajectory rid to ensure all requests goes to the same sglang server
        traj_rid = uuid.uuid4().hex
        while agent.num_turns < self.max_turns and not agent.is_finished:
            # The agent prepares the prompt and sampling params for LLM generation
            input_ids, sampling_params = agent.prepare_llm_query(self.tokenizer)

            # Send request to inference engine and get response
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            # agent extracts tool callings from the llm response
            tool_calls = agent.consume_llm_response(resp, completion_str)

            # call tool and compute reward
            if tool_calls is not None and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                res = (await self.toolbox.step((qid, [tool_call])))[0]
                agent.consume_tool_response(res, topk=self.topk)

                if "score" in res:
                    score = res["score"]
                if "ground_truth" in res:
                    ground_truth = res["ground_truth"]

            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        llm_gen_records = agent.memory.filter_records("llm_gen")
        format_reward = float(all([correct_format_fn(i, r.text) for i, r in enumerate(llm_gen_records)]))

        # compute rewards
        score = (score or 0) * format_reward
        pred_answer = agent.get_answer()
        judge_q_invalid = False
        if pred_answer is not None:
            judge_q_invalid = any([_c in pred_answer for _c in ["question", "invalid", "appropriate", "valid"]])
        if valid_inst and judge_q_invalid:
            score = -0.5
        
        stats = agent.memory.logging_stats()
        stats.update(dict(
            score=score,
            judge_q_invalid = judge_q_invalid,
            format_reward=format_reward,
        ))

        return ground_truth, score, agent.memory, stats       
    
    async def arun_episode(self, engine, data, epoch=None):
        # Get the unique identifier for this prompt
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex

        # 修复重复检测逻辑：只在训练时启用重复检测，评估时跳过
        if self.dump_dir is not None and epoch is not None and "generated-eval" not in self.dump_dir:
            # 只在训练目录中检查重复，跳过评估目录
            epoch_dir = os.path.join(self.dump_dir, f"epoch_{epoch}")
            if os.path.exists(epoch_dir):
                import glob
                # 搜索这个qid在当前epoch的所有步骤中
                pattern = os.path.join(epoch_dir, f"step_*", f"{qid}.jsonl")
                existing_files = glob.glob(pattern)

                if len(existing_files) > 0:
                    logger.info(f"[DEBUG] {qid} is already trained in epoch {epoch} - skipping")
                    return None

        # Initialize and Prepare the prompt
        version = engine.get_version()
        prompt_template = SEARCH_ONLY_PROMPT_TEMPLATE if self.search_only else SEARCH_ACCESS_PROMPT_TEMPLATE
        prompt = prompt_template.format(question=data["question"])
        valid_inst: bool = np.random.uniform(0, 1) <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)

        # Apply access strict ratio decay based on current version (global_step)
        if not self.search_only:
            current_step = version - 1  # version is global_step + 1
            access_strict_ratio = self.get_access_strict_ratio(current_step)
            use_strict_access = np.random.uniform(0, 1) <= access_strict_ratio

            # 打印衰减信息
            logger.info(f"[ACCESS DECAY] Step {current_step}: strict_ratio={access_strict_ratio:.3f}, use_strict={use_strict_access}")

            if use_strict_access:
                logger.info(f"[ACCESS DECAY] Using STRICT_ACCESS prompt at step {current_step}")
                prompt = prompt.replace(NORMAL_ACCESS, STRICT_ACCESS)
            else:
                logger.info(f"[ACCESS DECAY] Using NORMAL_ACCESS prompt at step {current_step}")

        prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Collect trajectories 
        trajs = await asyncio.gather(*[self.collect_agent_trajectory(valid_inst, qid, prompt, prompt_token_ids, engine) for _ in range(self.n_trajs)])

        ground_truth, scores, results, stats = None, [], [], []
        for gt, score, traj, traj_stats in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)
        
        raw_scores = scores
        score_mean = np.asarray(scores).mean()
        scores = [s-score_mean for s in scores]
        logger.info(f"Scores @ qid={qid}: {raw_scores} -> {scores}")
        # if all([s==0 for s in scores]):
        #     return None

        trajs = [traj for _, _, traj, _ in trajs]
        for i, traj_memory in enumerate(trajs):
            seqs = []
            for j, record in enumerate(traj_memory.memory):
                if record.type != "llm_gen":
                    continue

                # Check whether any previous seq is equivalent to input tokens
                success = False
                for seq in seqs:
                    if record.input_len  < len(seq["input_ids"]):
                        continue
                    h_cur = hash(record.input_tokens[:len(seq["input_ids"])])
                    h_seq = hash(seq["input_ids"])
                    if h_cur == h_seq:
                        seq_len = len(seq["input_ids"])
                        seq["input_ids"] = record.input_tokens + record.output_tokens
                        seq["logprobs"] += [0.0] * (record.input_len - seq_len) + record.output_logprobs
                        seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * record.output_len
                        seq["versions"] += [-1] * (record.input_len - seq_len) + record.output_versions
                        success = True
                        break
                if not success:
                    seq = dict(
                        input_ids = record.input_tokens + record.output_tokens,
                        logprobs = [0.0] * record.input_len + record.output_logprobs,
                        loss_mask = [0] * record.input_len + [1] * record.output_len,
                        versions = [-1] * record.input_len + record.output_versions,
                    )
                    seqs.append(seq)

            traj_stats = stats.pop(0)
            first_llm_gen = True
        
            for seq in seqs:
                res = dict(
                    # unsqueeze to add an additional batch dimension
                    input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
                    loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
                    logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
                    versions=torch.tensor(seq["versions"]).unsqueeze(0),
                    attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
                    # reward
                    rewards=torch.tensor([float(scores[i])]),
                )

                res.update(dict(begin_of_trajectory=torch.tensor([int(first_llm_gen)]),))
                res.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                first_llm_gen = False

                results.append(TensorDict(res, batch_size=[1]))

        if self.dump_dir is not None and epoch is not None:
            # 使用传入的epoch参数，而不是计算出来的epoch，确保一致性
            version = engine.get_version()
            global_step = max(0, version - 1)  # Ensure global_step is not negative
            step_calc = global_step % self.steps_per_epoch if self.steps_per_epoch else 0

            # 对于评估，使用global_step避免覆盖；对于训练，使用epoch/step结构
            if "generated-eval" in self.dump_dir:
                # 评估模式：使用 global_step 避免覆盖
                eval_dir = os.path.join(self.dump_dir, f"global_step_{global_step}")
            else:
                # 训练模式：使用 epoch/step 结构
                eval_dir = os.path.join(self.dump_dir, f"epoch_{epoch}", f"step_{step_calc}")

            os.makedirs(eval_dir, exist_ok=True)

            # Dump rollout to file
            file_path = os.path.join(eval_dir, f"{qid}.jsonl")
            with open(file_path, "w") as f:
                for i, (traj_memory, raw_score) in enumerate(zip(trajs, raw_scores)):
                    f.write(json.dumps(dict(memory=traj_memory.to_dict(), reward=raw_score, ground_truth=ground_truth, traj_idx=i)) + "\n")

        results = concat_padded_tensors(results)
        return results

@dataclass
class AgentRLConfig(GRPOConfig):
    max_turns: int = field(
        default=128,
        metadata={
            "help": "maximum number of turns for search agent"
        }
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        }
    )
    search_client_type: str = field(
        default="async-online-search-access",
        metadata={
            "help": "Type of tool (async-online-search-access/async-search-access). By default we use 'async-online-search-access'"
        }
    )
    reward_type: str = field(
        default="F1",
        metadata={
            "help": "The type of reward function"
        }
    )
    topk: int = field(
        default=5,
        metadata={
            "help": "search returns the top-k results. Default top_k=5"
        }
    )
    valid_inst_ratio: float = field(
        default=1.0,
        metadata={
            "help": "We randomly force a ratio of queries to produce valid anwers. By default valid_inst_ratio=1.0"
        }
    )
    search_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to use search only mode (no access). By default search_only=False to enable both search and access"
        }
    )
    # Logging Agent Trajectories
    log_agent_stats:  bool = field(
        default=False,
        metadata={
            "help": "Log stats for agent trajectories"
        },
    )
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: ["num_llm_gens"],
        metadata={
            "help": "Keys of log stats for agent trajectories"
        },
    )
    use_queue: bool = field(
        default=False,
        metadata={
            "help": "Whether to use queue for search requests"
        }
    )
    redis_config: dict = field(
        default_factory=dict,
        metadata={
            "help": "Redis configuration for queue"
        }
    )
    access_strict_start_ratio: float = field(
        default=0.9,
        metadata={
            "help": "Starting ratio for strict access prompt (0.0-1.0)"
        }
    )
    access_strict_end_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Ending ratio for strict access prompt after decay (0.0-1.0)"
        }
    )
    access_strict_decay_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps to decay from start_ratio to end_ratio"
        }
    )



def get_search_dataset(dataset_path, tokenizer, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    config: AgentRLConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    ref = None

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_search_dataset(config.train_dataset.path, tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )

    # Create evaluation dataloader
    if hasattr(config, 'valid_dataset') and config.valid_dataset.path:
        eval_dataloader = StatefulDataLoader(
            get_search_dataset(config.valid_dataset.path, tokenizer, rank, world_size),
            batch_size=config.valid_dataset.batch_size // world_size,
            shuffle=config.valid_dataset.shuffle,
            num_workers=config.valid_dataset.num_workers,
            collate_fn=lambda x: x,
            drop_last=False,  # 评估时保留不完整的batch，避免数据丢失
        )
    else:
        eval_dataloader = None

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Define steps_per_epoch for workflow
    steps_per_epoch = len(train_dataloader)

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # Create weight update meta for distributed training
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(AllocationMode.from_str(config.allocation_mode))

    actor.initialize(None, ft_spec)

    # Connect actor to inference engine (critical for weight updates) - must be after initialize
    actor.connect_engine(rollout, weight_update_meta)

    ref = None

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = SearchGymWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        dataset_path=config.train_dataset.path,
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
        search_only=getattr(config, 'search_only', False),
        use_queue=getattr(config, 'use_queue', False),
        redis_config=getattr(config, 'redis_config', {}),
        steps_per_epoch=steps_per_epoch,
        access_strict_start_ratio=getattr(config, 'access_strict_start_ratio', 0.9),
        access_strict_end_ratio=getattr(config, 'access_strict_end_ratio', 0.1),
        access_strict_decay_steps=getattr(config, 'access_strict_decay_steps', 100),
    )

    # Create evaluation workflow
    eval_workflow = SearchGymWorkflow(
        gconfig=config.gconfig.new(temperature=0.6),  # Lower temperature for evaluation
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
        dataset_path=config.valid_dataset.path if hasattr(config, 'valid_dataset') and config.valid_dataset.path else config.train_dataset.path,
        max_turns=32,
        n_trajs=1,  # Single trajectory for evaluation
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=1.0,  # Always valid for evaluation
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
        search_only=getattr(config, 'search_only', False),
        use_queue=getattr(config, 'use_queue', False),
        redis_config=getattr(config, 'redis_config', {}),
        steps_per_epoch=steps_per_epoch,  # Add steps_per_epoch for evaluation
        # For evaluation, we always use normal access (no strict access)
        access_strict_start_ratio=0.0,
        access_strict_end_ratio=0.0,
        access_strict_decay_steps=1,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # Recover
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow, epoch=epoch)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow, epoch=epoch)
            batch = tensor_container_to(batch, actor.device)
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )

        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            if config.log_agent_stats:
                agent_denominator = (batch["begin_of_trajectory"] > 0).bool()
                stats_tracker.denominator(agent=agent_denominator)
                stats_tracker.stat(
                    **{k: batch[k].float() for k in config.log_agent_stats_keys},
                    denominator="agent",
                )

            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("actor update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            print(f"[DEBUG VERSION] global_step={global_step}, setting version to {global_step + 1}")
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)
        
        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )
        
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):
            # def evaluate_fn():
            #     # 这个函数将在所有4个进程上执行
            #     rank = dist.get_rank()
            #     world_size = dist.get_world_size()

            #     print(f"[DEBUG EVAL] Rank {rank}/{world_size} entering evaluate_fn.")

            #     # ============================
            #     #  关键修复点：初始化时就指定 device
            #     # ============================
            #     local_stats = {
            #         "total_score": torch.tensor(0.0, device=actor.device),
            #         "total_format_reward": torch.tensor(0.0, device=actor.device),
            #         "total_samples": torch.tensor(0.0, device=actor.device, dtype=torch.float32)
            #     }

            #     # 确保eval_dataloader存在且其包含的数据集非空
            #     if eval_dataloader is not None and len(eval_dataloader.dataset) > 0:
            #         print(f"[DEBUG EVAL] Rank {rank} has {len(eval_dataloader.dataset)} samples. Starting evaluation on its split...")

            #         # 1. 提交所有本地评估任务
            #         cnt = 0
            #         for data in eval_dataloader:
            #             for item in data:
            #                 eval_rollout.submit(item, eval_workflow)
            #                 cnt += 1

            #         print(f"[DEBUG EVAL] Rank {rank} submitted {cnt} items. Waiting for results...")

            #         # 2. 等待并收集本地评估结果
            #         if cnt > 0:
            #             results_td = eval_rollout.wait(cnt, timeout=None)
            #             print(f"[DEBUG EVAL] Rank {rank} received results for {len(results_td['score'])} samples.")
            #             results_td = tensor_container_to(results_td, actor.device)
            #             # 3. 在本地聚合指标
            #             local_stats["total_score"] = results_td["score"].sum()
            #             local_stats["total_format_reward"] = results_td["format_reward"].sum()
            #             local_stats["total_samples"] = torch.tensor(len(results_td["score"]), device=actor.device, dtype=torch.float32)
            #         else:
            #             print(f"[WARN EVAL] Rank {rank} has an eval_dataloader but it yielded 0 items.")

            #     # 为了高效通信，将所有本地统计数据打包到一个tensor中
            #     # 此时，无论if分支是否执行，local_stats中的张量都在正确的GPU设备上
            #     stats_to_reduce = torch.stack([
            #         local_stats["total_score"],
            #         local_stats["total_format_reward"],
            #         local_stats["total_samples"]
            #     ])

            #     # 4. 使用 all_reduce 进行分布式聚合
            #     dist.all_reduce(stats_to_reduce, op=dist.ReduceOp.SUM)

            #     # 5. 仅在主进程（Rank 0）上计算并打印最终结果
            #     if actor.is_data_parallel_head():
            #         global_total_score = stats_to_reduce[0].item()
            #         global_total_format_reward = stats_to_reduce[1].item()
            #         global_total_samples = stats_to_reduce[2].item()

            #         print(f"\n--- [Rank 0] Distributed Evaluation Summary ---")

            #         if global_total_samples > 0:
            #             avg_score = global_total_score / global_total_samples
            #             avg_format_reward = global_total_format_reward / global_total_samples

            #             print(f"Total samples evaluated across all ranks: {int(global_total_samples)}")
            #             print(f"Average Score: {avg_score:.4f}")
            #             print(f"Average Format Reward: {avg_format_reward:.4f}")
            #         else:
            #             print("[WARN EVAL] No samples were evaluated across all ranks.")

            #         print(f"--------------------------------------------------\n")

            #     # 这个屏障至关重要
            #     dist.barrier(device_ids=[actor.device.index])
            #     current_platform.synchronize()

            # evaluator.evaluate(
            #     evaluate_fn,
            #     epoch,
            #     step,
            #     global_step,
            # )
            def evaluate_fn():
                # 这个函数将在所有进程上执行
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                
                print(f"[DEBUG EVAL] Rank {rank}/{world_size} entering evaluate_fn.")

                # 初始化本地统计数据，确保在正确的设备上
                local_stats = {
                    "total_score": torch.tensor(0.0, device=actor.device),
                    "total_format_reward": torch.tensor(0.0, device=actor.device),
                    "total_samples": torch.tensor(0.0, device=actor.device, dtype=torch.float32)
                }

                # 确保 eval_dataloader 存在且其包含的数据集非空
                if eval_dataloader is not None and len(eval_dataloader.dataset) > 0:
                    print(f"[DEBUG EVAL] Rank {rank} has {len(eval_dataloader.dataset)} samples. Starting evaluation on its split...")
                    
                    # 1. 提交所有本地评估任务
                    cnt = 0
                    for data in eval_dataloader:
                        for item in data:
                            # 直接提交 item，后续从返回结果中获取 qid
                            eval_rollout.submit(item, eval_workflow, epoch=epoch)  # 传递 epoch 参数
                            cnt += 1
                    
                    print(f"[DEBUG EVAL] Rank {rank} submitted {cnt} items. Waiting for results...")
                    
                    # 2. 等待并处理本地评估结果
                    if cnt > 0:
                        # 假设 results_td 是一个包含 CPU 张量的普通字典
                        results_td = eval_rollout.wait(cnt, timeout=None)
                        print(f"[DEBUG EVAL] Rank {rank} received results for {len(results_td.get('score', []))} samples.")
                        
                        # --- 使用 tensor_container_to 将数据移动到 GPU 进行计算 ---
                        results_td_gpu = tensor_container_to(results_td, actor.device)
                        
                        # --- 计算本地统计指标 (在GPU上) ---
                        local_stats["total_score"] = results_td_gpu["score"].sum()
                        local_stats["total_format_reward"] = results_td_gpu["format_reward"].sum()
                        local_stats["total_samples"] = torch.tensor(len(results_td_gpu["score"]), device=actor.device, dtype=torch.float32)

                        # --- eval_workflow会自动保存完整轨迹，无需在这里重复保存 ---
                        # eval_workflow的保存路径: eval_workflow.dump_dir = StatsLogger.get_log_path(config.stats_logger) + "/generated-eval"
                        # 保存格式: epoch_{epoch}/step_{step}/{qid}.jsonl (完整轨迹数据，和训练时一样)
                       

                        # 注释掉原来的保存逻辑，因为eval_workflow的arun_episode会自动保存
                        # # a. 准备保存目录 (所有进程使用同一个目录)
                        # eval_base_dir = os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated-eval")
                        # eval_dir = os.path.join(eval_base_dir, f"epoch_{epoch}_step_{step}")
                        #
                        # # b. 只有主进程负责创建目录，避免文件系统竞争
                        # if actor.is_data_parallel_head():
                        #     os.makedirs(eval_dir, exist_ok=True)
                        # # c. 所有进程等待，确保目录已创建
                        # dist.barrier(device_ids=[actor.device.index])
                        #
                        # print(f"[INFO] Rank {rank} is saving {len(results_td['score'])} traces to: {eval_dir}")
                        #
                        # # d. 此时 results_td 已经在 CPU 上，无需再次转换
                        # # 直接使用它进行文件操作
                        # results_td_cpu = tensor_container_to(results_td, "cpu")
                        #
                        # # e. 逐一保存每个轨迹
                        # for i in range(len(results_td_cpu['score'])):
                        #     # 从字典中提取第 i 个样本的所有信息
                        #     trace_data = {key: value[i].tolist() if hasattr(value[i], 'tolist') else value[i] for key, value in results_td_cpu.items()}
                        #
                        #     # 直接使用原始的 qid/id 作为文件名
                        #     qid = trace_data.get("qid", trace_data.get("query_id", trace_data.get("id", f"rank{rank}_fallback_{i}")))
                        #
                        #     # 构建要保存的对象
                        #     save_obj = {
                        #         "epoch": epoch,
                        #         "global_step": global_step,
                        #         "eval_type": "validation",
                        #         "generated_by_rank": rank, # 保留 rank 信息用于调试
                        #         **trace_data
                        #     }
                        #
                        #     # 文件名直接使用 qid
                        #     trace_file = os.path.join(eval_dir, f"{qid}.json")
                        #     with open(trace_file, "w") as f:
                        #         json.dump(save_obj, f, indent=2)
                        #
                        # print(f"[SUCCESS] Rank {rank} finished saving its traces.")
                        print(f"[SUCCESS] Rank {rank} - complete trace data handled by eval_workflow.arun_episode saving logic.")

                    else:
                        print(f"[WARN EVAL] Rank {rank} has an eval_dataloader but it yielded 0 items.")

                # --- 聚合全局统计指标 (所有进程都参与) ---
                stats_to_reduce = torch.stack(list(local_stats.values()))
                dist.all_reduce(stats_to_reduce, op=dist.ReduceOp.SUM)

                # --- 仅在主进程打印全局汇总信息 ---
                if actor.is_data_parallel_head():
                    global_total_score = stats_to_reduce[0].item()
                    global_total_format_reward = stats_to_reduce[1].item()
                    global_total_samples = stats_to_reduce[2].item()

                    print("\n--- [Rank 0] Distributed Evaluation Summary ---")
                    if global_total_samples > 0:
                        avg_score = global_total_score / global_total_samples
                        avg_format_reward = global_total_format_reward /                global_total_samples
                        print(f"Total samples evaluated across all ranks: {int              (global_total_samples)}")
                        print(f"Average Score: {avg_score:.4f}")
                        print(f"Average Format Reward: {avg_format_reward:.4f}")
                    else:
                        print("[WARN EVAL] No samples were evaluated across all                 ranks.")
                        avg_score = 0.0
                        avg_format_reward = 0.0
                    print("--------------------------------------------------\n")


                # 所有进程在此同步，确保文件写入和统计都完成后再继续
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

                
            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(stats_tracker.export_all(reduce_group=actor.parallelism_group))
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
