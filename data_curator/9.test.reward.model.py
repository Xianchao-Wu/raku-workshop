from openai import OpenAI
import os

# 👉 用你的 NVIDIA API Key（建议提前 export NVIDIA_API_KEY=xxx）
# use your NVIDIA API Key（need to export NVIDIA_API_KEY=xxx beforehand）
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NV_API_KEY")
)

model_name = "nvidia/llama-3.1-nemotron-70b-reward"

# 输入 input
question = "What is the capital of France?"
response = "The capital of France is Paris."

# 最简单 simplest prompt
prompt = f"""
Question: {question}
Response: {response}

Give a score between 0 and 1.
Only output the number.
"""

# 同步调用（没有 await） sync calling without await
completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.0,
    max_tokens=10,
)

# 输出结果 output result
text = completion.choices[0].message.content.strip()
print("Raw output:", text)

# 转成分数 to score (reward)
try:
    score = float(text)
    print("Score:", score)
except:
    print("Parse failed")
