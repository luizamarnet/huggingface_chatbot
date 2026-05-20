import warnings
warnings.filterwarnings('ignore')

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

huggingfacehub_api_token = os.getenv("OPENAI_API_KEY")
## 1. LLM (Hugging Face version)
def get_llm():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        max_new_tokens=256,
        temperature=0.5,
        #huggingfacehub_api_token=os.getenv(huggingfacehub_api_token)
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

## 2. Document loader (Stays the same)
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## 3. Text splitter (Stays the same)
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    chunks = [c for c in chunks if c.page_content.strip()]
    return chunks

## 4. Embedding model (Hugging Face version)
def hf_embedding():
    # 'all-MiniLM-L6-v2' is fast, free, and runs locally or via API
    # It is a direct competitor to IBM's Slate model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

## 5. Vector db (Updated to use HF embeddings)
def vector_database(chunks):
    embedding_model = hf_embedding()
    # Note: Chroma will create an in-memory db here
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## 6. Retriever (Stays the same)
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

## 7. QA Chain (Updated to use HF LLM)
def retriever_qa(file, query):

    llm = get_llm()

    retriever_obj = retriever(file)
    
    prompt_template = """<s>[INST] Answer based on the file received:
    {retriever_obj} [/INST]
    Question: {question}. Do not try to make up the answer.
    Include the pages and paragraphs of the respective pages where you found the information.
    """
    prompt = PromptTemplate.from_template(prompt_template)
   

    rag_chain = (
        {
            "retriever_obj": retriever_obj, # O retriever busca o texto relevante
            "question": RunnablePassthrough() # Passa a query adiante
        }
        | prompt
        | llm 
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    

    return response
