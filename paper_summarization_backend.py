import warnings
warnings.filterwarnings('ignore')

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA
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
def retriever_title(file):
    query = '''What is the title of this paper?
    Answer based on the file received. Do not try to make up the answer.
    If you do not have enough information, just you could not extract this information from the file '''
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)

    return response['result']

def retriever_authors(file):
    query = '''What are the names of the authors of this paper?
    Answer based on the file received. Do not try to make up the answer.
    If you do not have enough information, just you could not extract this information from the file '''
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)

    return response['result']

def retriever_topic(file):
    query = '''What is this paper about?
    Answer based on the file received. Do not try to make up the answer.
    If you do not have enough information, just you could not extract this information from the file.
    Include the pages and paragraphs of the respective pages where you found the information '''
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)

    return response['result']

def retriever_conclusions(file):
    query = '''What is this paper conclusions?
    Answer based on the file received. Do not try to make up the answer.
    If you do not have enough information, just you could not extract this information from the file.
    Include the pages and paragraphs of the respective pages where you found the information '''
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=False)
    response = qa.invoke(query)

    return response['result']

def retriever_qa(file):
    
    title = retriever_title(file)
    authors = retriever_authors(file)
    topic = retriever_topic(file)
    conclusions = retriever_conclusions(file)

    response = [title, authors, topic, conclusions]

    return response