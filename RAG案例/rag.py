




from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        self.vector_service=VectorStoreService(
           embedding=DashScopeEmbeddings(model=config.embeddings_model_name),
        )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                (
                 "system","以我提供的参考资料为主,"
                    "简洁和专业的回答用户问题.参考资料:{context}."),
                ("system","并且我提供的用户的对话历史记录,如下:"),
                MessagesPlaceholder("history"),
                ("user","请用户回答提问:{input}")
            ]
        )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=self.__get_chain()

    def __get_chain(self):
        retrieve=self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str =""
            for doc in docs:
                formatted_str+=f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str
        def temp1(value:dict) -> str:
            return value["input"]

        def temp2(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain=(
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(temp1)|retrieve | format_document
            } |RunnableLambda(temp2)| self.prompt_template|print_prompt|self.chat_model|StrOutputParser()
        )
        conversation_chain=RunnableWithMessageHistory(
            chain,

            get_history,
            input_messages_key="input",  # 输入键名
            history_messages_key="history"
        )
        return conversation_chain
if __name__=="__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    res=RagService().chain.invoke({"input":"针织毛衣如何让保养"},session_config)
    print(res)