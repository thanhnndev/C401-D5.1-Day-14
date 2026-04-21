import os
from openai import OpenAI

# Note: The base_url varies by region. The following example uses the base_url for the Singapore region.
# - Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# - US (Virginia): https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - China (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1
# - China (Hong Kong): https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1
# - Germany (Frankfurt): https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1. Replace {WorkspaceId} with your workspace ID.
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY").strip("'\""), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
completion = client.chat.completions.create(
    model="qwen2.5-vl-72b-instruct",
    messages=[{"role": "user", "content": "Who are you?"}]
)
print(completion.choices[0].message.content)