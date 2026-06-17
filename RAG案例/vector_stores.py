from langchain_chroma import Chroma
from openai.types import embedding

import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
if __name__ == "__main__":
    from  langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    res=retriever.invoke("我的体重160斤,尺码推荐")
    print(res)