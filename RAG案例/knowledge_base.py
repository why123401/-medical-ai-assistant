

import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_md5(md5_str:str):

    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()
            if line==md5_str:
                return True
        return False



def save_md5(md5_str:str):
    with open(config.md5_path,'a',encoding='utf-8') as f:
     f.write(md5_str+'\n')

def get_string_md5(input_string:str,encoding='utf-8'):
    # 将字符串转换为bytes字节数组
    str_bytes = input_string.encode(encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma=Chroma(
            collection_name = config.collection_name,
            embedding_function = DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory = config.persist_directory,
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self,data:str,filename):
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "跳过内容已经存在知识库中"
        if len(data)>config.max_split_char_number:
            knowledge_chunks: list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]
        metadata = {
            "source":filename,
            # "create_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小刘",
            }
        self.chroma.add_texts(knowledge_chunks,
                              metadata=  [metadata for _ in knowledge_chunks],
                              )
        save_md5(md5_hex)
        return "ok"
if __name__ == "__main__":
    service: KnowledgeBaseService = KnowledgeBaseService()
    r=service.upload_by_str("周2","text")
    print(r)
