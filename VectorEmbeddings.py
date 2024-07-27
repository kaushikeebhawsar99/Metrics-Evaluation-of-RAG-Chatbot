from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os, warnings
import streamlit as st
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

loader = CSVLoader(file_path="myntra_products_catalog.csv", encoding='utf-8')
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Load environment variables from .env file
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")


# Convert texts to embeddings
try:
  embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
except Exception as e:
  print(f"Error creating embeddings: {e}")

    # Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")
# Add documents to the vector store
vector_store.add_documents(documents=texts)

print("Embeddings generated and stored successfully.")