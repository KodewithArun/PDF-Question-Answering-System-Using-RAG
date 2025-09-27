from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
