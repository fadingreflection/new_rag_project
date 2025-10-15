"""Ml model loader for token generation. Russian language used for current task."""
from torch import float16
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

MODEL_ID = "unsloth/Qwen2.5-3B-unsloth-bnb-4bit" # неплохо работает на русском, нужный размер эмбеддингов 768
TASK_TYPE = "text-generation"

class MlModelLoader:
    def __init__(self, model_id: str, task_type: str):
        self.model_id = model_id
        self.task_type = task_type
        self.bnb_configs = self.set_bnb_configs()
        self.model = self.load_llm()
        self.tokenizer = self.load_tokenizer()

    def set_bnb_configs(self):
        # Квантуем в 4 бита, чтобы поместилось в VRAM 6–8 ГБ
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=float16        # тип данных для вычислений
            )
        return bnb_config

    def load_llm(self):
        if self.bnb_configs is None:
            self.bnb_configs = self.set_bnb_configs()
        #TODO @fadingreflection change to logs
        print("Загружаем модель …")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=self.bnb_configs,
            device_map="auto",
            trust_remote_code=True,
            )
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return tokenizer

    def build_pipe(self):
        llm_pipeline = pipeline(
            self.task_type,
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=float16,
            )
        return llm_pipeline

