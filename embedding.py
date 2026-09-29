import shutil

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"
MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

@st.cache_resource
def charger_embeddings():
    # Modèle recommandé sur le tuto Huggingface - Tsy voatery omena token, fa raha misy token dia faster downloads + higher rate limits
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Vectorisation et stockage dans une base de donnée vectorielle
# Ici j'utilise Chromadb
def vectoriser(chunks):
    # supprime tout l'ancien bdd
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = charger_embeddings()
    base_vectorielle = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return base_vectorielle

#Fonction d'extraction des fichiers
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
        # elif suffix.lower() == ".txt" or suffix.lower() == ".md":
        #     loader = TextLoader(chemin_tmp, encoding="utf-8")
        # Pour tout autres sources de text
        else:
            loader = TextLoader(chemin_tmp, encoding="utf-8")

        docs = loader.load()

        # Mettre dans metadata["source"] le nom du vrai fichier original
        # pour donner la source d'où vient cette extrait de texte.
        for doc in docs:
            doc.metadata["source"] = fichier.name

        documents.extend(docs)
        os.remove(chemin_tmp)  # nettoyage

    return documents


def chunk_basic(documents):
    """
    Fonction de Chunking recursive par characters
    La fonction va separer le document avec des chunk de taille maximale de 1000 chars
    La fonction va d'abord séparer le contenu du document par paragraphe paragraphe \n\n si le chunk a une taille>1000 chars
    La fonction va séparer par ligne \n. Ensuite, si le chunk a encore une taille > 1000 chars
    La fonction va séparer par phrase ". ", "! ", "? ", puis par mot (à chaque espace)
    et en dernier recours on découpe char par char.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return splitter.split_documents(documents)


def chunk(documents: list[Document]) -> list[Document]:
    """
    Fonction de Chunking finale
    Utilisation de chunk basic
    Mais découpe selon le format : les fichiers markdown (.md) par section markdown utilisant markdown_splitter
    Puis après utilise chunk_basic pour split les section trop grande (supérieur a chunk_size)
    """

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "subtitle"),
            ("###", "section"),
        ],
        strip_headers=True,
    )

    chunks_final = []

    for document in documents:
        source = document.metadata.get("source", "")
        extension = Path(source).suffix.lower()

        if extension in {".md", ".markdown"}:
            sections = markdown_splitter.split_text(
                document.page_content
            )

            for section in sections:
                # Conserver les métadonnées du fichier d'origine.
                section.metadata = {
                    **document.metadata,
                    **section.metadata,
                    "type": "markdown_section",
                }

                # Split seulement si la section est trop grande.
                section_chunks = chunk_basic([section])

                for section_chunk in section_chunks:
                    chunks_final.append(section_chunk)

        else:
            # Split basic pour les PDF et autres txt.
            document.metadata["type"] = (
                "pdf" if extension == ".pdf" else "text"
            )

            chunks_final.extend(chunk_basic([document]))

    return chunks_final

