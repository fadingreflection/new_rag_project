"""VectorDB builder."""

from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# DATA_ID = "Den4ikAI/russian_cleared_wikipedia"
# SPLITTER = "train"
# RECORD_KEY = "sample"
MODEL_ID_SS = "intfloat/multilingual-e5-base"
DB_DIR = "chroma_ragmini"

class VectorDBBuilder:
    def __init__(self, data_id: str, splitter: str, record_key: str):
        self.data_id = data_id
        self.splitter = splitter
        self.record_key = record_key
        self.data = self.load_data_from_hub()

    def load_data_from_hub(self):
        data = load_dataset(self.data_id, split=self.splitter)
        corpus_docs = [
            Document(page_content=rec[self.record_key])
            for rec in data
        ]
        # TODO: @fadingreflection change to logs
        print("Документов:", len(corpus_docs))
        print(corpus_docs[0].page_content[:200], "…")
        return corpus_docs

    def load_custom_data(self):
        pass

    def get_chunks(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            )
        docs = splitter.split_documents(self.data)
        #TODO @fadingreflection change to logs
        print("Чанков:", len(docs))
        return docs

    def get_embeds(self):
        embeddings=HuggingFaceEmbeddings(
            model_name=MODEL_ID_SS, #Russian semantic search
            model_kwargs={"device": "cuda"},
        )
        return embeddings

    def get_vector_db(self): #добавить проверку непустого хранилища 
        from pathlib import Path
        path_obj = Path(DB_DIR)
        if path_obj.exists() and path_obj.is_dir() and any(path_obj.iterdir()):
            vectordb = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.get_embeds()
                )
        else:
            vectordb = Chroma.from_documents(
                documents=self.get_chunks(),               # либо corpus_docs, если без сплиттера
                embedding=self.get_embeds(),
                persist_directory=DB_DIR  # директория для хранения векторной базы
                )
            vectordb.persist() # в нашем рабочем пространстве создалась директория - векторное хранилище
        return vectordb
