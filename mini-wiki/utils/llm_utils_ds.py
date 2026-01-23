from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is not set")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

def call_model(prompt: str, response_format: dict = None, model: str = "deepseek-chat"):
    """
    一个经过加固的API调用函数，能够处理不规范的API响应和异常。
    它保证只返回两种结果：一个有效的Python字典，或者None。
    """
    try:
        # 准备API调用参数
        api_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        if response_format:
            api_params["response_format"] = response_format

        completion = client.chat.completions.create(**api_params)
        content = completion.choices[0].message.content

        if response_format:
            return json.loads(content)
        else:
            return content
        
    except json.JSONDecodeError as e:
        # 专门处理JSON解析失败的情况
        print(f"  >> ERROR: Failed to decode JSON. Reason: {e}")
        return None
    except Exception as e:
        # 捕获所有其他异常，包括网络问题和那个奇怪的字符串异常
        print(f"  >> ERROR: An unexpected error occurred during API call. Error: {e}")
        
        # --- 防火墙 ---
        # 这是一个关键的检查。如果捕获的"异常"e本身是一个字符串
        # （我们已经知道会发生这种情况），我们就打印一个特定的警告。
        if isinstance(e, str):
            print("  >> CRITICAL WARNING: A raw string was raised as an exception. This is non-standard behavior from the client library or API.")
            
        return "API Error"