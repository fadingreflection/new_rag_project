"""Custom prompt template to use as answer schema."""

from langchain_core.prompts import PromptTemplate

simple_custom_prompt = PromptTemplate(
    template="""<think>
Сначала тщательно подумай, потом ответь на заданный вопрос. Избегай повторений. Будь краток, если не попросят пояснений. Для короткого ответа достаточно одного- двух предложений.

Context: {context}
Question: {question}

</think>
<answer>""",
    input_variables=["context", "question"]
)
