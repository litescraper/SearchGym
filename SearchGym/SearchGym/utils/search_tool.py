# Copyright 2025 Ant Group Inc.
import json
from typing import List, Tuple

from realhf.base import logging
from SearchGym.utils.rewards import compute_score_em, compute_score_f1

from SearchGym.utils.search_utils import make_search_client

logger = logging.getLogger("Search ToolBox")

def load_metadata(dataset_path):
    data=[json.loads(ff) for ff in open(dataset_path)]
    for d in data:
        if "idx" in d:
            d["idx"] = str(d["idx"])
        elif "qid" in d:
            d["idx"] = str(d["qid"])
        else:
            d["idx"] = str(d["id"])
    id2info = {d["idx"]: d for d in data}
    return id2info


class SearchToolBox:
    def __init__(self, dataset_path: str, reward_type: str = "F1", topk:int = 10, search_client_type: str = "async-online-search-access", use_jina=False, redis_config: dict = None, use_queue: bool = False, search_only: bool = False):
        self.id2info = load_metadata(dataset_path)
        self.reward_type = reward_type
        self.topk = topk

        # search server
        self.use_jina = use_jina
        self.search_client_type = search_client_type
        self.redis_config = redis_config
        self.use_queue = use_queue
        self.search_only = search_only
        self.search_client = None
    
    async def initialize_search_client(self):
        """Initialize search client with Redis support"""
        if self.search_client is None:
            if self.search_client_type == "redis-queued-search":
                self.search_client = await make_search_client(
                    self.search_client_type,
                    use_jina=self.use_jina,
                    jina_api_key=None,
                    redis_config=self.redis_config
                )
            elif self.search_client_type == "meilisearch-local" and self.use_queue:
                # 为Meilisearch启用队列模式
                from meilisearch_client import make_meilisearch_client
                self.search_client = make_meilisearch_client(
                    use_queue=self.use_queue,
                    redis_url=self.redis_config.get("url", "redis://localhost:6379")
                )
            else:
                self.search_client = make_search_client(
                    self.search_client_type,
                    use_jina=self.use_jina,
                    jina_api_key=None
                )

    
    async def step(self, qid_actions: Tuple[str, List[str]]):
        qid, actions = qid_actions

        # Initialize search client if needed
        await self.initialize_search_client()

        results = []
        for action in actions:
            result = dict(documents=None, score=None, ground_truth=None, type=None)

            # tool calling
            if "<search>" in action and "</search>" in action:
                query = action.split("<search>")[-1].split("</search>")[0].strip()
                req_meta = {
                    "queries": [query],
                    "topk": self.topk,
                    "return_scores": False
                }

                # send search query to server
                response = await self.search_client.query_async(req_meta)
                documents = response[0]["documents"]
                urls = response[0]["urls"]

                result["documents"] = documents
                result["urls"] = urls
                # 传递完整数据和搜索模式信息
                if "full_contents" in response[0]:
                    result["full_contents"] = response[0]["full_contents"]
                if "titles" in response[0]:
                    result["titles"] = response[0]["titles"]
                result["search_only"] = self.search_only  # 传递search_only模式
                result["type"] = "search"
            elif "<access>" in action and "</access>" in action:
                url = action.split("<access>")[-1].split("</access>")[0].strip()

                # send wepage access request
                response = await self.search_client.access_async([url])
                page = None

                if self.search_client_type == "async-online-search-access":
                    if self.use_jina:
                        page = response[0].get("page", "")
                    else:
                        # process webpage
                        page = self.process_webpage(response[0].get("page", ""))
                elif self.search_client_type == "async-search-access":
                    if response[0] is None:
                        page = None
                    else:
                        page = response[0].get("page", "")
                elif self.search_client_type == "meilisearch-local":
                    if response and len(response) > 0:
                        page = response[0].get("page", "")
                    else:
                        page = None

                result["page"] = page
                result["type"] = "access"

            # compute rewards
            ground_truth = self.id2info[qid.split("@")[0]]["answer"]
            if isinstance(ground_truth, list) or isinstance(ground_truth, tuple):
                ground_truth = [str(gt) for gt in ground_truth]
            else:
                ground_truth = str(ground_truth)

            ground_truth_aug = None
            if "aug_answer" in self.id2info[qid.split("@")[0]] and len(self.id2info[qid.split("@")[0]]["aug_answer"]) > 0:
                ground_truth_aug = self.id2info[qid.split("@")[0]]["aug_answer"]
                if isinstance(ground_truth_aug, list) or isinstance(ground_truth_aug, tuple):
                    ground_truth_aug = [str(gt) for gt in ground_truth_aug]
                else:
                    ground_truth_aug = str(ground_truth_aug)
            
            if self.reward_type == "F1":
                extracted, score = compute_score_f1(action, ground_truth, method="strict")
            elif self.reward_type == "EM":
                extracted, score = compute_score_em(action, ground_truth, method="strict")
            if ground_truth_aug is not None:
                if self.reward_type == "F1":
                    _, score_aug = compute_score_f1(action, ground_truth_aug, method="strict")
                elif self.reward_type == "EM":
                    _, score_aug = compute_score_em(action, ground_truth_aug, method="strict")

            result["extracted"] = extracted
            result["score"] = score
            result["ground_truth"] = self.id2info[qid.split("@")[0]]["answer"]
            
            if ground_truth_aug is not None:
                score_aug = max(score_aug, score)
                result["score"] = score * 0.7 + score_aug * 0.3
                result["ground_truth_aug"] = ground_truth_aug
                
            # if extracted is not None:
            #     logger.info("F1 Score={:.2f}. Extracted='{}'. Ground Truth='{}'. Qid={}. Question='{}'".format(score, extracted, ground_truth, qid.split("@")[0], self.id2info[qid.split("@")[0]]["question"]))

            results.append(result)
        return results

    def process_webpage(self, content):
        keys = [("title", "title"), ("p", "p"), ("li", "li", lambda c: "\n" not in c), ("td", "td"), ("tr", "tr")] 
        content_list = []
        init_length = len(content)
        while any([f"<{k[0]}" in content and f"</{k[1]}>" in content for k in keys]):
            klr = []
            for k in keys:
                start = 0
                # print(k)
                while True:
                    ls = [content[start:].find(f"<{k[0]}{c}") for c in [">", " "]]
                    ls = [l for l in ls if l != -1]
                    l = -1 if len(ls) == 0 else min(ls)
                    # print(ls)
                    if l == -1:
                        break
                    l += start
                    r = content[l:].find(f"</{k[1]}>")
                    if r == -1:
                        break
                    if (len(k) <= 2) or (len(k) >= 3 and k[2](content[l:l+r])):
                        # print(k, l, l+r)
                        klr.append((k, l, l+r))
                        break
                    start = l + r

            if len(klr) == 0:
                break
            klr = sorted(klr, key=lambda x:x[1])
            k, l, r = klr[0]
            content_list.append(content[l:r+len(f"</{k[1]}>")])
            # print(content_list[-1])
            # input("stop...")
            if k[0] == "p":
                content_list[-1] += "\n\n"
            elif k[0] == "li":
                content_list[-1] += "\n"
            content = content[r:]
        content = "".join(content_list)
        final_length = len(content)
        logger.info(f"process the webpage: {init_length} -> {final_length}. {content[:100]}")
        return content
