import aiohttp
import asyncio
import json
import os
import uuid
import time
from typing import Dict, Any, List, Optional
import random
try:
    import redis.asyncio as redis_async
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available, queue functionality will be limited")


class MeiliSearchClient:
    """本地MeiliSearch客户端，包装成与SearchGym相同的接口，支持内存队列"""

    def __init__(self, meilisearch_url: str = "http://127.0.0.1:7700",
                 index_name: str = "wiki",
                 api_key: str = "aSampleMasterKey",
                 use_queue: bool = False,
                 redis_url: str = "redis://localhost:6379"):
        self.meilisearch_url = meilisearch_url
        self.index_name = index_name
        self.api_key = api_key
        self.search_url = f"{meilisearch_url}/indexes/{index_name}/search"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        # 存储搜索结果的完整内容，用于access
        self._search_cache = {}  # url -> full_content

        # 队列功能
        self.use_queue = use_queue
        self.redis_url = redis_url
        self.redis_client = None
        self._redis_available = False

        # 内存队列（主要队列机制）
        self._memory_request_queue = asyncio.Queue(maxsize=5000)  # 增加队列容量
        self._memory_result_queues = {}  # request_id -> asyncio.Queue
        self._worker_tasks = []
        self._running = False

        # 队列配置
        self.request_timeout = 180  # 减少超时时间到3分钟
        self.poll_timeout = 10      # 减少轮询超时到10秒

        # 初始化队列
        if self.use_queue:
            asyncio.create_task(self._initialize_queue())
    def _generate_url(self, title: str) -> str:
        """
        生成一个复杂的、无实际意义的URL。
        
        Args:
            title: 来自搜索结果的原始标题。
            
        Returns:
            一个格式化后的URL，例如：https://data.storage.net/entries/Machine_Learning
        """
        # 随机选择协议
        protocol = random.choice(["http", "https"])
        
        # 随机选择一些无意义的域名
        domain = random.choice([
            "web.content.provider",
            "data.storage.net",
            "info.retriever.org",
            "docs.repository.local",
            "digital.archive.io",
            "global.resource.info",
            "the-public-index.com",
            "secure.records.vault",
            "universal.asset.host",
            "query.service.online",
            "shared.data.center",
            "virtual.document.store",
            "main.data-source.net",
            "primary.info.gateway"
        ])
        
        # 随机选择一些路径段
        path_segment = random.choice(["articles", "documents", "pages", "entries"])
        
        # 将标题中的空格替换为下划线，以符合URL规范
        sanitized_title = title.replace(" ", "_")
        
        # 组合成最终的URL
        return f"{protocol}://{domain}/{path_segment}/{sanitized_title}"

    async def _initialize_queue(self):
        """初始化内存队列"""
        print("使用内存队列处理搜索请求")

        # 启动工作线程
        self._running = True
        self._worker_tasks = [
            asyncio.create_task(self._search_worker())
            for _ in range(16)  # 16个工作线程处理队列，提高并发能力
        ]
        print(f"Started {len(self._worker_tasks)} search workers (内存队列)")

    async def _push_to_queue(self, queue_name: str, data: Dict):
        """推送数据到队列"""
        if queue_name == "search_requests":
            await self._memory_request_queue.put(data)
        elif queue_name.startswith("results:"):
            request_id = queue_name[8:]  # Remove "results:" prefix
            if request_id not in self._memory_result_queues:
                self._memory_result_queues[request_id] = asyncio.Queue()
            await self._memory_result_queues[request_id].put(data)
        else:
            await self._memory_request_queue.put(data)

    async def _pop_from_queue(self, queue_name: str, timeout: int = 30) -> Optional[Dict]:
        """从队列中获取数据"""
        try:
            if queue_name == "search_requests":
                result = await asyncio.wait_for(self._memory_request_queue.get(), timeout=timeout)
            elif queue_name.startswith("results:"):
                request_id = queue_name[8:]  # Remove "results:" prefix
                if request_id not in self._memory_result_queues:
                    self._memory_result_queues[request_id] = asyncio.Queue()
                result = await asyncio.wait_for(self._memory_result_queues[request_id].get(), timeout=timeout)
            else:
                result = await asyncio.wait_for(self._memory_request_queue.get(), timeout=timeout)

            return result
        except asyncio.TimeoutError:
            return None

    async def _search_worker(self):
        """搜索工作线程，处理队列中的搜索请求"""
        while self._running:
            try:
                # 从队列获取请求
                request_data = await self._pop_from_queue("search_requests", timeout=1)
                if request_data is None:
                    continue

                request_id = request_data["request_id"]
                queries = request_data["queries"]
                topk = request_data["topk"]

                # print(f"Processing search request {request_id} with {len(queries)} queries")

                # 顺序处理每个查询
                for i, query in enumerate(queries):
                    try:
                        # 执行MeiliSearch搜索
                        results = await self._search_single(query, topk)

                        # 构造结果消息
                        result_message = {
                            "request_id": request_id,
                            "query_index": i,
                            "query": query,
                            "results": results,
                            "is_last": (i == len(queries) - 1),
                            "status": "success"
                        }

                    except Exception as e:
                        print(f"Search error for query '{query}' in request {request_id}: {e}")
                        result_message = {
                            "request_id": request_id,
                            "query_index": i,
                            "query": query,
                            "error": str(e),
                            "is_last": (i == len(queries) - 1),
                            "status": "error"
                        }

                    # 推送到结果队列
                    await self._push_to_queue(f"results:{request_id}", result_message)

                # print(f"Completed search request {request_id}")

            except Exception as e:
                print(f"Search worker error: {e}")
                await asyncio.sleep(1)

    async def _search_single(self, query: str, topk: int = 10) -> Dict:
        """执行单个搜索查询"""
        try:
            payload = {
                "q": query,
                "limit": topk
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.search_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    meili_result = await response.json()

                    # 转换为SearchGym格式
                    hits = meili_result.get("hits", [])
                    documents = []
                    urls = []
                    full_contents = []  # 完整内容
                    titles = []        # 标题列表

                    for hit in hits:
                        title = hit.get("title", "")
                        abstract = hit.get("abstract", "") or title
                        full_content = hit.get("content", "")

                        documents.append(abstract)
                        titles.append(title)
                        full_contents.append(full_content)

                        # 直接使用MeiliSearch返回的URL，如果有的话
                        url = hit.get("url", "")
                        if not url:
                            # 如果没有URL，回退到生成URL
                            url = self._generate_url(title)
                        urls.append(url)

                        # 缓存完整内容（保持向后兼容）
                        self._search_cache[url] = full_content

                    return {
                        "documents": documents,
                        "urls": urls,
                        "full_contents": full_contents,  # 新增：完整内容
                        "titles": titles,               # 新增：标题列表
                        "server_type": "meilisearch-queue",
                    }

        except Exception as e:
            print(f"MeiliSearch search error for '{query}': {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Search URL: {self.search_url}")
            import traceback
            traceback.print_exc()
            return {
                "documents": [],
                "urls": [],
                "server_type": "meilisearch-queue",
            }

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        """
        执行搜索查询，返回abstract和url，存储完整内容用于后续access
        如果启用队列模式，会使用Redis队列顺序处理搜索请求

        Args:
            req_meta: 搜索请求元数据，包含queries, topk等字段

        Returns:
            格式化后的搜索结果列表
        """
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)

        if not queries:
            return []

        # 如果使用队列模式
        if self.use_queue:
            return await self._query_with_queue(queries, topk)

        # 普通模式
        results = []

        for query in queries:
            try:
                # 构建MeiliSearch请求
                payload = {
                    "q": query,
                    "limit": topk
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.search_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        response.raise_for_status()
                        meili_result = await response.json()

                        # 转换MeiliSearch结果为SearchGym格式
                        hits = meili_result.get("hits", [])

                        documents = []
                        urls = []
                        full_contents = []  # 完整内容
                        titles = []        # 标题列表

                        for hit in hits:
                            title = hit.get("title", "")
                            abstract = hit.get("abstract", "") or title
                            full_content = hit.get("content", "")

                            documents.append(abstract)
                            titles.append(title)
                            full_contents.append(full_content)

                            # 直接使用MeiliSearch返回的URL，如果有的话
                            url = hit.get("url", "")
                            if not url:
                                print("Warning: Hit missing URL, generating one")
                                url = f"http://wiki/{title.replace(' ', '_')}"
                            urls.append(url)

                            # 缓存完整HTML内容用于access（保持向后兼容）
                            self._search_cache[url] = full_content

                        query_result = {
                            "documents": documents,
                            "urls": urls,
                            "full_contents": full_contents,  # 新增：完整内容
                            "titles": titles,               # 新增：标题列表
                            "server_type": "meilisearch-local",
                        }
                        results.append(query_result)

            except Exception as e:
                print(f"MeiliSearch query error for '{query}': {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Search URL: {self.search_url}")
                import traceback
                traceback.print_exc()
                # 返回空结果
                results.append({
                    "documents": [],
                    "urls": [],
                    "server_type": "meilisearch-local",
                })

        return results
    
    async def access_async(self, urls: List[str]) -> List[Dict]:
        """
        访问URL内容，返回完整的HTML内容
        
        Args:
            urls: 要访问的URL列表
            
        Returns:
            访问结果列表，包含完整HTML内容
        """
        if not urls:
            return []
        
        results = []
        for url in urls:
            # 从缓存中获取完整内容
            full_content = self._search_cache.get(url, "")
            
            # 如果缓存中没有内容，尝试从wiki.json文件中查找
            if not full_content:
                full_content = self._find_content_in_wiki_json(url)
            
            result = {
                "page": full_content,  # 返回完整的HTML内容
                "type": "access",
                "server_type": "meilisearch-local",
            }
            results.append(result)
        
        return results
    
    def _find_content_in_wiki_json(self, url: str) -> str:
        """
        从wiki.json文件中查找对应title的content内容

        Args:
            url: 格式为 ".../{title}" 的URL或完整的URL

        Returns:
            找到的content内容，如果没找到返回空字符串
        """
        try:
            # 从URL中提取title
            if not url:
                return ""

            # 使用 split('/')[-1] 可以简单有效地获取路径的最后一部分
            encoded_title = url.split('/')[-1]
            title = encoded_title.replace("_", " ")

            # 构造wiki.json文件路径
            wiki_json_path = "/home/xzhangjq/mini-wiki/outputs/wiki/wiki_with_urls.json"
            
            # 检查文件是否存在
            if not os.path.exists(wiki_json_path):
                return ""
            
            # 读取wiki.json文件
            with open(wiki_json_path, 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
            
            # 优先通过URL直接匹配
            if isinstance(wiki_data, list):
                for item in wiki_data:
                    if isinstance(item, dict):
                        # 首先尝试直接匹配URL
                        if item.get("url") == url:
                            return item.get("content", "")
            elif isinstance(wiki_data, dict):
                # 如果是字典格式，尝试不同的键名
                if title in wiki_data:
                    return wiki_data[title].get("content", "")
                elif "articles" in wiki_data and isinstance(wiki_data["articles"], list):
                    for item in wiki_data["articles"]:
                        if isinstance(item, dict):
                            if item.get("url") == url:
                                return item.get("content", "")
            
            return ""
            
        except Exception as e:
            print(f"Error finding content in wiki.json for '{url}': {e}")
            return ""

    async def _query_with_queue(self, queries: List[str], topk: int) -> List[Dict]:
        """使用队列处理搜索请求"""
        # 等待队列初始化完成
        if not self._redis_available and not self._memory_request_queue:
            # 如果队列不可用，降级为直接处理
            return await self._process_immediately(queries, topk)

        # 生成唯一请求ID
        request_id = str(uuid.uuid4())

        # 提交请求到队列
        request_data = {
            "request_id": request_id,
            "queries": queries,
            "topk": topk,
            "timestamp": time.time()
        }

        await self._push_to_queue("search_requests", request_data)
        # print(f"Queued search request {request_id} with {len(queries)} queries")

        # 收集结果
        all_results = []
        expected_results = len(queries)
        collected_results = 0

        start_time = time.time()
        while collected_results < expected_results:
            # 检查超时
            if time.time() - start_time > self.request_timeout:
                print(f"Timeout waiting for results from request {request_id}")
                break

            # 获取下一个结果
            result = await self._pop_from_queue(f"results:{request_id}", timeout=self.poll_timeout)

            if result is None:
                # 没有结果，继续等待
                continue

            if result.get("status") == "error":
                print(f"Error in search result: {result.get('error')}")
                # 添加空结果
                all_results.append({
                    "documents": [],
                    "urls": [],
                    "server_type": "meilisearch-queue"
                })
            else:
                # 添加结果
                search_result = result.get("results", {})
                all_results.append(search_result)

            collected_results += 1

            # 检查是否最后一个结果
            if result.get("is_last", False):
                break

        # print(f"Collected {collected_results}/{expected_results} results for request {request_id}")

        # 填充缺失的结果
        while len(all_results) < expected_results:
            all_results.append({
                "documents": [],
                "urls": [],
                "server_type": "meilisearch-queue"
            })

        return all_results

    async def _process_immediately(self, queries: List[str], topk: int) -> List[Dict]:
        """立即处理搜索请求（内存模式或降级模式）"""
        results = []
        for query in queries:
            try:
                search_result = await self._search_single(query, topk)
                results.append(search_result)
                await asyncio.sleep(0.1)  # 查询间小延迟
            except Exception as e:
                print(f"Error in immediate processing for '{query}': {e}")
                results.append({
                    "documents": [],
                    "urls": [],
                    "server_type": "meilisearch-queue"
                })
        return results

    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        if not self.use_queue:
            return {"queue_enabled": False}

        queue_length = 0
        if self._redis_available and self.redis_client:
            try:
                queue_length = await self.redis_client.llen("search_requests")
            except:
                pass

        return {
            "queue_enabled": True,
            "redis_available": self._redis_available,
            "queue_length": queue_length,
            "workers_running": len([t for t in self._worker_tasks if not t.done()]),
            "memory_queue_size": self._memory_request_queue.qsize()
        }

    async def close(self):
        """关闭客户端，清理资源"""
        self._running = False

        # 取消工作线程
        for task in self._worker_tasks:
            task.cancel()

        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        # 关闭Redis客户端
        if self.redis_client:
            await self.redis_client.close()


# 为了与SearchGym的make_search_client函数兼容，添加一个工厂函数
def make_meilisearch_client(meilisearch_url: str = "http://localhost:7700",
                            index_name: str = "wiki",
                            api_key: str = "aSampleMasterKey",
                            use_queue: bool = False,
                            redis_url: str = "redis://localhost:6379"):
    """创建MeiliSearch客户端实例"""
    return MeiliSearchClient(
        meilisearch_url,
        index_name,
        api_key,
        use_queue=use_queue,
        redis_url=redis_url
    )


if __name__ == "__main__":
    # 测试MeiliSearch客户端
    async def test_meilisearch():
        print("=== Testing MeiliSearch Client ===")

        # 测试普通模式
        print("\n--- Testing Normal Mode ---")
        client = MeiliSearchClient()

        req_meta = {
            "queries": ["machine learning"],
            "topk": 2,
            "return_scores": False
        }

        results = await client.query_async(req_meta)
        print(f"Normal mode results: {len(results)} queries processed")

        # 测试队列模式
        print("\n--- Testing Queue Mode ---")
        queue_client = MeiliSearchClient(use_queue=True)

        # 等待队列初始化
        await asyncio.sleep(1)

        # 检查队列状态
        status = await queue_client.get_queue_status()
        print(f"Queue status: {status}")

        # 测试队列搜索
        req_meta = {
            "queries": ["artificial intelligence", "neural networks"],
            "topk": 2,
            "return_scores": False
        }

        results = await queue_client.query_async(req_meta)
        print(f"Queue mode results: {len(results)} queries processed")

        # 测试访问
        if results and results[0]['urls']:
            urls = results[0]['urls'][:1]  # 只测试第一个URL
            access_results = await queue_client.access_async(urls)
            print(f"Access results: {len(access_results)} URLs accessed")

        # 清理
        await queue_client.close()

    # 运行测试
    asyncio.run(test_meilisearch())