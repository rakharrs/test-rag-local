import shutil

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

CHROMA_DIR = "chroma_db"

@st.cache_resource
def charger_embeddings():
    # Modèle recommandé sur le tuto Huggingface - Tsy voatery omena token, fa raha misy token dia faster downloads + higher rate limits
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#Vectorisation
def vectoriser(chunks):
    # supprime tout l'ancien bdd
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    # Chargement de l'embedding hf into mettre dans Chromadb
    embeddings = charger_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore

def extract(fichiers):
    #Sauvegarde les fichiers uploadés sur disque et extrait leur contenu.
    documents = []

    for fichier in fichiers:
        # Écriture temporaire sur disque (nécessaire pour les loaders LangChain)
        suffix = os.path.splitext(fichier.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(fichier.getvalue())
            chemin_tmp = tmp.name

        # Choix du loader selon l'extension
        if suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(chemin_tmp)
        else:  # .txt, .md
            loader = TextLoader(chemin_tmp, encoding="utf-8")

        docs = loader.load()

        # Nom du vrai fichier original
        for doc in docs:
            doc.metadata["source"] = fichier.name

        documents.extend(docs)
        os.remove(chemin_tmp)  # nettoyage

    return documents

#Chunking
def chunk(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)