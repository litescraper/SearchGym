#!/usr/bin/env python3
"""
为wiki.json中的每个条目生成一个唯一的URL
"""

import json
import random
import sys

def generate_url(title: str) -> str:
    """
    生成一个复杂的、无实际意义的URL。

    Args:
        title: 来自搜索结果的原始标题。

    Returns:
        一个格式化后的URL，例如：https://data.storage.net/entries/Machine_Learning
    """
    # 随机选择协议
    protocol = random.choice(["http", "https"])

    # 随机选择域名
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

def main():
    """主函数"""
    # 设置随机种子以确保可重现性（可选）
    random.seed(42)

    # 读取wiki.json文件
    wiki_json_path = "/home/xzhangjq/mini-wiki/outputs/wiki/wiki.json"

    print(f"正在读取 {wiki_json_path}...")

    try:
        with open(wiki_json_path, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print(f"✅ 成功读取 {len(wiki_data)} 条wiki数据")

    # 为每个条目生成URL
    processed_count = 0

    for item in wiki_data:
        if 'title' in item:
            # 生成URL
            url = generate_url(item['title'])

            # 添加URL到条目中
            item['url'] = url

            processed_count += 1

            # 显示进度
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 条记录...")

    print(f"✅ 成功为 {processed_count} 条记录生成URL")

    # 显示一些示例
    print("\n📋 示例生成的URL:")
    for i, item in enumerate(wiki_data[:5]):
        print(f"  {i+1}. {item['title']} -> {item['url']}")

    # 保存更新后的数据
    output_path = "/home/xzhangjq/mini-wiki/outputs/wiki/wiki_with_urls.json"

    print(f"\n正在保存到 {output_path}...")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(wiki_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功保存到 {output_path}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return

    # 验证URL唯一性
    urls = [item.get('url') for item in wiki_data if 'url' in item]
    unique_urls = set(urls)

    print(f"\n📊 URL统计:")
    print(f"  总URL数: {len(urls)}")
    print(f"  唯一URL数: {len(unique_urls)}")
    print(f"  唯一性: {len(unique_urls) == len(urls)}")

    if len(unique_urls) == len(urls):
        print("✅ 所有URL都是唯一的!")
    else:
        print(f"⚠️  发现 {len(urls) - len(unique_urls)} 个重复URL")

if __name__ == "__main__":
    main()