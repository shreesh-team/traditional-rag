from fastapi import FastAPI
from langchain_ollama import ChatOllama


app = FastAPI()


# ability to upload files
# whole rag implementation using lcel
# create a rag class
# each class function does the required step
# get the final result in response

# v0.01
# direct query to llm using RAG without document upload
# v0.1
# ability to upload document and query


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/queryllm")
async def queryllm(query: str):
    ollama_llm = ChatOllama(model="gemma3:12b", temperature=0)
    response = ollama_llm.invoke(query)
    return {"response": response.content}
