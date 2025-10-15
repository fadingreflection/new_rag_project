"""VectorDB builder."""

from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.vector_db.custom_doc_processor import CustomDocProcessor

MODEL_ID_SS = "intfloat/multilingual-e5-base"
DB_DIR = "chroma_ragmini"
DB_DIR_CUSTOM = "my_personal_docs_db"
CUSTOM_DOC_PATH = r"C:\Users\Anastasiya Fedotova\Desktop\квартирный вопрос\Аванс Ак. Пилюгина фин.docx"
LOAD_VIA_LANGCHAIN_FLAG = False

class VectorDBBuilder:
    def __init__(self, data_id: str, splitter: str, record_key: str, use_custom_data=False, recreate_db=False):
        self.data_id = data_id
        self.splitter = splitter
        self.record_key = record_key
        self.use_custom_data_flag = use_custom_data
        self.recreate_db_flag = recreate_db
        self.data = self.load_data_from_hub() if not self.use_custom_data_flag else self.load_custom_data()
        self.database_directory = DB_DIR_CUSTOM if self.use_custom_data_flag else DB_DIR

    def load_data_from_hub(self):
        data = load_dataset(self.data_id, split=self.splitter)
        corpus_docs = [
            Document(page_content=rec[self.record_key])
            for rec in data
        ]
        # TODO: @fadingreflection change to logs
        print("Документов:", len(corpus_docs))
        # print(corpus_docs[0].page_content[:200], "…")
        return corpus_docs

    def load_custom_data(self):
        custom_proc_inst = CustomDocProcessor(custom_doc_path=CUSTOM_DOC_PATH,
                                              load_via_langchain=LOAD_VIA_LANGCHAIN_FLAG)
        documents, is_langchain_format = custom_proc_inst.custom_proc_pipe()
        if is_langchain_format:
            return documents
        else:
            documents = self.custom_text_splitter(documents)
            return documents

    def custom_text_splitter(self, clean_text):
        #TODO @fadingreflection move to global params to be set
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]  # Приоритет разделителей
        )
        chunks = text_splitter.split_text(clean_text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        return docs

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
        path_obj = Path(self.database_directory)
        if path_obj.exists() and path_obj.is_dir() and any(path_obj.iterdir()) and not self.recreate_db_flag:
            vectordb = Chroma(
                persist_directory=self.database_directory,
                embedding_function=self.get_embeds()
                )
        else:
            vectordb = Chroma.from_documents(
                documents=self.get_chunks(),               # либо corpus_docs, если без сплиттера
                embedding=self.get_embeds(),
                persist_directory=self.database_directory # директория для хранения векторной базы
                )
            vectordb.persist() # в нашем рабочем пространстве создалась директория - векторное хранилище
        return vectordb
