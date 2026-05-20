import warnings
warnings.filterwarnings('ignore')

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import unicodedata
import os


load_dotenv()

huggingfacehub_api_token = os.getenv("OPENAI_API_KEY").strip()
## 1. LLM (Hugging Face version)
def get_llm():

    model_id = "deepseek-ai/DeepSeek-R1"   #"meta-llama/Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"   #meta-llama/Meta-Llama-3-8B-Instruct 
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation", #"conversational",
        max_new_tokens=512,
        temperature=0.1,
        #huggingfacehub_api_token=os.getenv(huggingfacehub_api_token)
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    return llm

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
            c.page_content = f"{fixed_text}\n\n[SOURCE: This text is strictly from Page {page_num} of the PDF]"
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
    # Force Chroma to initialize a completely clean, non-persistent registry on every click
    ephemeral_client = chromadb.EphemeralClient()
    
    # Note: Chroma will create an in-memory db here
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=ephemeral_client  # Binds execution exclusively to this live transaction
    )
    
    return vectordb

'''## 6. Retriever (Stays the same)
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever(search_kwargs={"k": 8})
'''

## 7. QA Chain (Updated to use HF LLM)
def retriever_qa(file):
    
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_obj = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = get_llm()
    

    # ESTRATÉGIA PARA AUTORES: Lê diretamente o início do documento
    primeira_pagina = splits[0].page_content[:2000] # Pega os primeiros 2000 caracteres do PDF
    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_autores = f"""Analise o texto do topo de um artigo científico abaixo e extraia APENAS o título do artigo.
    Do not try to make up the answer.
    Return only the title of the article or a phrase saying you could not extract the names from  the text.
    Texto do Artigo:
    {primeira_pagina}
    Title:"""
    # Invoca o modelo de chat diretamente
    authors_res = llm.invoke(prompt_autores)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    title = authors_res if isinstance(authors_res, str) else authors_res.content


    ##############

    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_title = f"""Analise o texto do topo de um artigo científico abaixo e extraia APENAS os nomes dos autores que escreveram este paper.
    Os nomes dos autores do paper geralmente estão localizados logo abaixo do título principal no contexto fornecido.
    Retorne apenas os nomes, de forma direta, separados por virgula e com um ponto no final. O ultimo e penúltimo autores devem ser separados pela palavra e
    Do not try to make up the answer.
    If you don't find the answer, return a phrase saying you could not extract the names from  the text.
    Texto do Artigo:
    {primeira_pagina}
    Autores:"""
    # Invoca o modelo de chat diretamente
    authors_res = llm.invoke(prompt_title)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    authors = authors_res if isinstance(authors_res, str) else authors_res.content

    # Usamos o LLM diretamente (sem passar pelo RetrievalQA) para extrair os nomes
    prompt_affiliation = f"""Analise o texto do topo de um artigo científico abaixo e extraia APENAS as afiliações dos autores.
    As afiliações geralmente estão localizadas próximas aos nomes dos autores dos artigos e ou conectados a esses nomes por caracteres sobreescritos ou sobescritos.
    Retorne apenas as afiliações, uma em cada linha.
    Caso junto com o nome da instituicao venha acompanhada da informacao dos paí, cidade ou estado onde a mesma se encontra (e nao como parte do nome da institucao), essas informacoes de localidade devem vir entre parenteses. 
    Do not try to make up the answer.
    If you don't find the answer, return a phrase saying you could not extract the names from  the text.
    Note: The input text layer might contain slight character encoding artifacts (e.g., messed up accents or symbols in names). Automatically correct these names to their standard academic spelling if you recognize them.
    Texto do Artigo:
    {primeira_pagina}
    Afiliações (e.g. ChaBot University (Country)):"""
    # Invoca o modelo de chat diretamente
    affiliation_res = llm.invoke(prompt_affiliation)
    # Dependendo da versão, use: authors_output = authors_res.content se retornar um objeto de mensagem
    affiliation = affiliation_res if isinstance(affiliation_res, str) else affiliation_res.content


    ##############
    
    template = """You are an AI assistant specialized in analyzing scientific papers.
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

    Question: {question}
    Answer with page and section citations:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # Note: The input text layer might contain slight character encoding artifacts (e.g., messed up accents or symbols in names). Automatically correct these names to their standard academic spelling if you recognize them.
    
    topic_res = qa.invoke("What is the main topic of this paper? State the core objective and contributions")
    topic = topic_res["result"]
    
    conclusions_res = qa.invoke("What are the main conclusions of this study? State the primary takeaway of the conclusion ")
    conclusions = conclusions_res['result']

    response = [title, authors, affiliation, topic, conclusions]

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