import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
import glob
import os
import pinecone
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import os 
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


## Lets Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')## This contains the documents

## Lets chunk the data
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)# This contains the chunked documents

## Load the model for embeddings
embeddings = HuggingFaceEmbeddings()

import os 
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone() # Create a Pinecone client

# Create a new Pinecone vector index
import time
from langchain_pinecone import PineconeVectorStore

index_name = "text-embeddings"
namespace = "wondervector5000"
pc.create_index(
    name="text-embeddings",
    dimension=768, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


import time
from langchain_pinecone import PineconeVectorStore

namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=doc,
    index_name=index_name,
    embedding=embeddings, 
    namespace=namespace 
)

time.sleep(1)

#setup LLM model
from langchain_huggingface import llms, HuggingFaceEndpoint


# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
  repo_id=repo_id,
  huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
)


from langchain_core.prompts import PromptTemplate

template = """
You are a Game Reccommender Expert. These Human will ask you a questions about which game to play. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer more than 4 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)


from langchain.chains import RetrievalQA  
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )
