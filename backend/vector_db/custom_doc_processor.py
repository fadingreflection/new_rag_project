"""Custom document processor."""
import re

import docx
from langchain.document_loaders import UnstructuredWordDocumentLoader


class CustomDocProcessor:
    def __init__(self, custom_doc_path: str, load_via_langchain=False):
        self.path = custom_doc_path
        self.is_langchain_flag = load_via_langchain
        self.document = self.load_via_langchain() if self.is_langchain_flag else self.read_docx()

    def load_via_langchain(self):
        loader = UnstructuredWordDocumentLoader(self.path)
        documents = loader.load()
        return documents

    def read_docx(self):
        doc = docx.Document(self.path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Пропускаем пустые параграфы
                full_text.append(paragraph.text)
        return "\n".join(full_text)

    def clean_text(self):
        # Удаляем лишние переносы строк (но сохраняем абзацы)
        text = self.document
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[^\w\s.,!?;:()\-–—\n]', '', text)
        return text.strip()

    def custom_proc_pipe(self):
        if self.is_langchain_flag:
            documents = self.document
            return documents, self.is_langchain_flag
        clean_text = self.clean_text()
        return clean_text, self.is_langchain_flag
