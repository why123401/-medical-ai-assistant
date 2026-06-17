from  langchain_ollama import OllamaLLM
model=OllamaLLM(model = "qwen3-vl:4b")
res=model.invoke(input = "你是谁呀你能做什么")
print(res)
