"""Rag system constructor."""
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

from backend.nlp_pipeline.customize_prompt import simple_custom_prompt


class RagBuilder:
    def __init__(self, vectordb, llm_pipeline):
        self.retriver = vectordb.as_retriever()
        self.generator = HuggingFacePipeline(
            pipeline=llm_pipeline,
        )

    def build_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.generator,
            chain_type="stuff",
            retriever=self.retriver,
            chain_type_kwargs={"prompt": simple_custom_prompt},
            return_source_documents=True,
            )
        return qa_chain
