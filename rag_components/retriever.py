from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

def build_retriever(vectorstore, docs, child_splitter, parent_splitter):
    # Ensure docs are clean
    pure_docs = [doc if isinstance(doc, Document) else doc[0] for doc in docs]
    filtered_docs = filter_complex_metadata(pure_docs)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(filtered_docs)
    return retriever
