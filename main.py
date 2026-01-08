from fastapi import FastAPI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        topk=2,
        chunks=[],
        persist_directory: str = "./chroma_db",
    ):
        self.embeddings = OllamaEmbeddings(model="embeddinggemma")
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorDb = None
        self.topk = topk
        self.vectorDb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        self.retriever = self.vectorDb.as_retriever(
            search_type="similarity", search_kwargs={"k": self.topk}
        )

    def get_vector_db(self):
        return self.vectorDb

    def get_vector_retriever(self):
        return self.retriever

    def test_vector_retriever(self):
        test_query = "What is the main topic in this document"
        return self.retriever.invoke(test_query)


class TraditionalRAG:
    def __init__(self, embeddings, system_prompt, llm):
        self.embeddings = embeddings
        self.system_prompt = system_prompt
        self.prompt = ChatPromptTemplate.from_template(self.system_prompt)
        self.llm = llm

    # load the document
    # chunking of the document
    # use embedding models to create vectors of chunks
    # push vectors in vector db
    # retrieval from vector db
    # rag chain

    async def loadDocs(self, pdf_path):
        pdf_loader = PyPDFLoader(pdf_path)
        docs = pdf_loader.load()
        return docs

    async def chunkTheDocs(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
        return chunks

    async def test_embedding(self):
        sample_text = "This is a test sentence for embeddings"
        embedding = self.embeddings.embed_query(sample_text)
        return embedding

    async def get_retriever(self, retriever, format_docs):
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/queryllm")
async def queryllm(query: str):
    ollama_llm = ChatOllama(model="gemma3:12b", temperature=0)
    response = ollama_llm.invoke(query)
    return {"response": response.content}


@app.get("/rag")
async def rag(query: str, pdf_path: str):
    embeddings = OllamaEmbeddings(model="embeddinggemma")
    sys_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer concise and accurate.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )
    ollama_llm = ChatOllama(model="gemma3:12b", temperature=0)
    rag = TraditionalRAG(
        embeddings=embeddings, system_prompt=sys_prompt, llm=ollama_llm
    )
    docs = await rag.loadDocs(pdf_path)
    chunks = await rag.chunkTheDocs(docs)
    # test_embeddings = await rag.test_embedding()
    vector_store = ChromaVectorStore(
        collection_name="test_collection",
        chunks=chunks,
        topk=5,
    )
    retriever = vector_store.get_vector_retriever()
    # response = retriever.invoke(query)

    rag_chain = await rag.get_retriever(
        retriever=retriever,
        format_docs=format_docs,
    )

    rag_chain_response = rag_chain.invoke(query)

    return {"response": rag_chain_response}
