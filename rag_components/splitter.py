from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_child_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

def get_parent_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
