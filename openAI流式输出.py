from openai import OpenAI

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

)
# 2调用模型
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "你是一个python专家,并且话非常多"},
        {"role": "assistant", "content": "好的,我是一个编程专家,并且话非常多,你要问什么"},
        {"role": "user", "content": "输出1-10的数字,用python代码"}
    ],
    stream=True
)
for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end="\r",
        flush=True
    )
