"""RAG inference."""
import sys
from pathlib import Path
# Добавить родительскую директорию
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from backend.rag_constructor.rag_construct import RagBuilder
from backend.vector_db.db_builder import VectorDBBuilder
from backend.model_loader.loader import MlModelLoader

DATA_ID = "Den4ikAI/russian_cleared_wikipedia"
SPLITTER = "train"
RECORD_KEY = "sample"
MODEL_ID_SS = "intfloat/multilingual-e5-base"
DB_DIR = "chroma_ragmini"
MODEL_ID = "unsloth/Qwen2.5-3B-unsloth-bnb-4bit" # неплохо работает на русском, нужный размер эмбеддингов 768
TASK_TYPE = "text-generation"
MARKER = "</think>"

from torch import cuda
DEVICE = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
#TODO @fadingreflection change to logs
print(f"Current device is: {DEVICE}")

class RagModelInference:
    def __init__(self):
        self.marker = MARKER
        self.vectordb_builder_inst = VectorDBBuilder(data_id=DATA_ID,
                                                     splitter=SPLITTER,
                                                     record_key=RECORD_KEY)
        self.vectordb = self.vectordb_builder_inst.get_vector_db()
        self.model_loader_inst = MlModelLoader(model_id=MODEL_ID, task_type=TASK_TYPE)
        self.llm_pipe = self.model_loader_inst.build_pipe()
        self.rag_builder_inst = RagBuilder(self.vectordb, self.llm_pipe)

    def rag_system_inference(self):
        qa_chain = self.rag_builder_inst.build_chain()
        response = qa_chain.invoke("Коротко расскажи про разницу яблок антоновка и яблок сорта белый налив")
        print("Ответ:", response["result"].split(MARKER, 1)[1].strip())
        print("\nSource documents:")
        for doc in response["source_documents"]:
            print(doc.page_content[:80], "…")


inst = RagModelInference()
inst.rag_system_inference()