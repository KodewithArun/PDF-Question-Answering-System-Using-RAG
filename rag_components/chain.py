from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(retriever, prompt, llm):
    retriever_chain = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))
    
    format_inputs = RunnableLambda(lambda x: {
        "context": "\n\n".join([doc.page_content for doc in x["context"]]),
        "question": x["question"]
    })

    return RunnableSequence(
        {
            "question": RunnablePassthrough(),
            "context": retriever_chain
        },
        format_inputs,
        prompt,
        llm,
        StrOutputParser()
    )
