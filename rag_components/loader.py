from langchain_community.document_loaders import UnstructuredPDFLoader

def load_pdf(file_path: str):
    loader = UnstructuredPDFLoader(
        file_path=file_path,
        mode="paged",
        strategy="auto",
        encoding="utf-8",
    )
    return loader.load()
