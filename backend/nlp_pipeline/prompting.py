"""NLP prompt."""

CONFIG_DICT = {"max_new_tokens":200,     # генерируем ≤ 200 новых токенов
               "do_sample":True,         # позволяет модели додумывать
               "truncation":True,        # обрываем слишком длинные ответы
               "top_k":20,               # топ-20 наиболее вероятных токенов на выходе генерации
               "num_return_sequences":1, # 1 вариант ответа
               }

class AskLLM:
    def __init__(self, prompt: str, tokenizer, llm_pipeline, config_dict=CONFIG_DICT):
        self.prompt = prompt
        self.config_dict = config_dict
        self.tokenizer = tokenizer
        self.llm_pipeline = llm_pipeline

    def ask_qwen(self):
        resp = self.llm_pipeline(
            self.prompt,
            **CONFIG_DICT,
            num_return_sequences=1, # 1 вариант ответа
            eos_token_id=self.tokenizer.eos_token_id, # что считать концом последовательности
        )[0]["generated_text"]
        return resp
