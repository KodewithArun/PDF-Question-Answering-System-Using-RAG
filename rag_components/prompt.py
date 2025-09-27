from langchain.prompts import PromptTemplate

def get_chatbot_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant trained to answer questions from a PDF document.
Use only the information provided in the context below to answer the user's question.
If the answer is not present in the context, politely say you don't know.

Context:
{context}

User Question:
{question}

Answer in a clear and professional tone:
"""
    )
