from langchain_community.vectorstores import Chroma

def get_vectorstore(embedding, persist_directory="../chroma_db"):
    return Chroma(
        collection_name="pdf_chunks",
        embedding_function=embedding,
        persist_directory=persist_directory
    )
