import warnings
warnings.filterwarnings('ignore')

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
import unicodedata
import secrets
import os
import gc


load_dotenv()

huggingfacehub_api_token = os.getenv("OPENAI_API_KEY").strip()
## 1. LLM (Hugging Face version)
def get_llm():
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct" #"deepseek-ai/DeepSeek-R1"   #"meta-llama/Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"   #meta-llama/Meta-Llama-3-8B-Instruct 
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational", #"conversational",
        max_new_tokens=512,
        temperature=0.1,
        #huggingfacehub_api_token=os.getenv(huggingfacehub_api_token)
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model
    #return llm

## 2. Document loader (Stays the same)
def document_loader(file):
    loader = PyMuPDFLoader(file)
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
    #chunks = [c for c in chunks if c.page_content.strip()]
    clean_chunks = []
    for c in chunks:
        # Normalize text to NFKC format to repair broken accent mappings (like ą, ł, é, ü)
        fixed_text = unicodedata.normalize('NFKC', c.page_content)
        # Strip trailing white spaces and skip empty entries
        if fixed_text.strip():
            page_num = c.metadata.get("page", 0) + 1
            c.page_content = f"{fixed_text}\n[SOURCE: This text is strictly from Page {page_num} of the PDF]"
            clean_chunks.append(c)
    return clean_chunks

## 4. Embedding model (Hugging Face version)
def hf_embedding():
    # 'all-MiniLM-L6-v2' is fast, free, and runs locally or via API
    # It is a direct competitor to IBM's Slate model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

## 5. Vector db (Updated to use HF embeddings)
def vector_database(chunks):

    embedding_model = hf_embedding()
    
    # 1. Force Chroma to initialize a completely clean, non-persistent registry
    ephemeral_client = chromadb.EphemeralClient()
    
    # 2. Generate a totally unique collection name for this specific file run
    # This prevents the second run from conflicting with the "langchain" default name
    #unique_collection_name = f"paper_{secrets.token_hex(4)}"

    fixed_name = "current_paper_collection"
    # Safely clear out any lingering data with the same name in this session thread
    try:
        ephemeral_client.delete_collection(name=fixed_name)
    except Exception:
        pass # Collection didn't exist yet, safe to proceed
    
    
    # 3. Bind execution exclusively to this live transaction with a unique ID
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=ephemeral_client,
        collection_name=fixed_name # Keeps it fixed so Hugging Face can utilize its server-side routing cache
    )
    
    
    return vectordb

'''## 6. Retriever (Stays the same)
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever(search_kwargs={"k": 8})
'''

# Helper to convert document arrays into clean formatted strings for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## 7. QA Chain (Updated to use HF LLM)
def retriever_qa(file):
    
    # Create a completely random string to bypass Hugging Face's remote server caching
    cache_buster = secrets.token_hex(4)

    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = get_llm()
    
    # ESTRATÉGIA PARA AUTORES: Lê diretamente o início do documento
    first_page = splits[0].page_content[:4000] # Pega os primeiros 2000 caracteres do PDF
    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_title = f"""Analyze the text at the top of the scientific article below, and extract only the article's title.
    Do not try to make up the answer.
    Return only the title of the article or a phrase saying you could not extract this information from the text. Answer in English.
    Article's Text:
    {first_page}
    Title:"""
    # Invoca o modelo de chat diretamente
    title_res = llm.invoke(prompt_title)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    title = title_res if isinstance(title_res, str) else title_res.content


    ##############

    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_authors = f"""Analyze the text at the top of the scientific article below, and extract only the names of the authors who wrote this paper.
    The authors' names are usually located just below the main title in the given context.
    Return only the names, in a straightforward manner, separated by commas and with a period at the end. The second last and last authors should be separated by the word "and".
    Do not try to make up the answer.
    If you don't find the answer, return a phrase saying you could not extract the names from the text.
    Note: These names might contain slight character encoding artifacts (e.g., messed-up accents or symbols in names). Automatically correct these names to their standard academic spelling if you recognize them.
    Article Text:
    {first_page}
    Authors:"""
    # Invoca o modelo de chat diretamente
    authors_res = llm.invoke(prompt_authors)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    authors = authors_res if isinstance(authors_res, str) else authors_res.content

    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_affiliation = f"""Analyze the text at the top of the scientific article below, and extract only the authors' affiliations.
    Affiliations are usually located near the authors' names in the articles and/or connected to those names by superscript or subscript characters.
    Affiliations can also be in a paragraph which explains which is the institution (affiliation) connected to which author.
    Return only the affiliations, one per line.
    If the institution's name is accompanied by information about the country, city, or state where it is located (and not as part of the institution's name), this location information should be in parentheses.
    Do not try to make up the answer.
    If you don't find the answer, return a phrase saying you could not extract the affiliations from the text.
    Article's Text:
    {first_page}
    Affiliation (e.g. ChaBot University (Country)):"""
    # Invoca o modelo de chat diretamente
    affiliation_res = llm.invoke(prompt_affiliation)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    affiliation = affiliation_res if isinstance(affiliation_res, str) else affiliation_res.content


    ##############
    
    template = """You are an AI assistant specialized in analyzing scientific papers.
    [Session Tracker Security Token: {cache_id}]
    Using the following context excerpts retrieved from a scientific paper PDF, answer the final question.
    
    CRITICAL INSTRUCTIONS:
    1. Every chunck of context ends with a tag like '[SOURCE: Page X]'. You MUST read this tag and explicitly state the exact Page number and Section in your answer.
    2. Provide a direct, professional answer based on the facts, results, and discussions present in the text.
    3. If the exact information for the answer is split across several text chunks, synthesize them logically without adding external assumptions.
    4. Focus heavily on explicit paper abstract, goals, core methodologies, results, and conclusions.
    5. If you can infer the answer from results, do not write disclaimers about information not being explicit.
    6. Do not try to make up or guess information. If you could not retrieve the answer from the text, just say it.
    8. STRICT CONSTRAINT: For every fact you extract, you MUST provide a short 'Search phrase' verbatim quote. This Search phrase MUST BE EXACTLY BETWEEN 4 AND 5 WORDS. NEVER exceed 5 words per Search phrase.
    7. Answer examples: 
        (a) The model achieves X [Page 4, Section M](Search phrase: "Our empirical findings indicate")
        (b) The model achieves X [Page 4, Section M](Search phrases: "Our empirical findings indicate", "The gain percentage was")
        (b) The model achieves X [Page 4, Section M](Search phrases: "Our empirical findings indicate")[Page 7, Section N](Search phrases: "The gain percentage was")
    9. Answering using the formatting presented in bullet point 7 is mandatory.
    10. Remember this is a academic paper. Information with citations, which means, followed by a number inside parenthesis or with other authors names or attributed to other authors are not a goal, result or conclusion of the current paper.

    Context:
    {context}

    Question: {query}
    Answer with page and section citations:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "query", "cache_id"],
        template=template,
    )
    
    '''qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    input_key="query", 
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # Note: The input text layer might contain slight character encoding artifacts (e.g., messed up accents or symbols in names). Automatically correct these names to their standard academic spelling if you recognize them.
    '''

    # --- MODERNISED STATELESS LCEL CHAIN PIPELINE ---
    # This architecture natively allows routing cache_id and query safely to your prompt
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever_obj.invoke(x["query"])),
            "query": lambda x: x["query"],
            "cache_id": lambda x: x["cache_id"]
        }

        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    topic_res = rag_chain.invoke({"query": "What is the main topic of this paper? State the core objective and contributions",
                          "cache_id": cache_buster})
    topic = topic_res#["result"]
    
    conclusions_res = rag_chain.invoke({"query": "What are the main conclusions of this study? State the primary takeaway of the conclusion ",
                                "cache_id": cache_buster})
    conclusions = conclusions_res#['result']

    response = [title, authors, affiliation, topic, conclusions]

    '''# --- CRITICAL FIX 2: FORCE DESTRUCTION OF COMPILATION OBJECTS ---
    # Manually delete references to prevent memory retention across session runs
    #del qa
    del rag_chain
    del retriever_obj
    del vectordb
    del chunks
    del splits
    del llm
    
    # Execute full hardware garbage collection to flush memory blocks completely
    gc.collect()'''

    return response


'''from langchain_core.prompts import PromptTemplate

def retriever_qa(file):
    # 1. Cria a base de dados vetorial uma única vez
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever()
    
    llm = get_llm()

    # 2. Define a estrutura base do Prompt Template (Instrução do Sistema + Contexto + Pergunta)
    # O modelo DEVE receber as variáveis {context} e {question} exatamente com esses nomes
    template = """Você é um assistente de IA especializado em análise de artigos científicos.
Use os seguintes trechos de contexto recuperados do arquivo PDF para responder à pergunta no final.
Responda EXCLUSIVAMENTE em português.
Se você não sabe a resposta ou se o contexto não contiver a informação, diga explicitamente que não foi possível extrair essa informação do arquivo. Não invente dados.

Contexto:
{context}

Pergunta: {question}
Resposta em português:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 3. Executa as consultas passando o prompt customizado através do 'chain_type_kwargs'
    # Consulta 1: Título
    qa_title = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    title_res = qa_title.invoke("What is the title of this paper?")
    
    # Consulta 2: Autores
    qa_authors = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    authors_res = qa_authors.invoke("What are the names of the authors of this paper?")
    
    # Consulta 3: Tópico
    qa_topic = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    topic_res = qa_topic.invoke("What is this paper about? Include references to pages if found.")
    
    # Consulta 4: Conclusões
    qa_conclusions = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    conclusions_res = qa_conclusions.invoke("What are this paper's conclusions?")

    return [title_res['result'], authors_res['result'], topic_res['result'], conclusions_res['result']]
'''