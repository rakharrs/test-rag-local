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

st.set_page_config(
    page_title="Assistant RAG local",
    page_icon="📚",
    layout="wide",
)

st.title("Assistant RAG local")
# st.caption("Interrogez vos documents sans envoyer vos données vers une API externe.")


# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialisation de vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# Barre latérale
with st.sidebar:
    st.header("Documents")

    fichiers = st.file_uploader(
        "Ajouter des fichiers",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    llm_active = st.toggle(
        "Activer le LLM",
        value=False,
        help="Désactivé : recherche sémantique. Activé : assistant RAG.",
    )

    if st.button("Indexer les documents", use_container_width=True):
        if not fichiers:
            st.warning("Veuillez d'abord sélectionner un fichier.")
        else:
            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                documents = extract(fichiers)
                chunks = chunk(documents)
                vectorstore = vectoriser(chunks)
                st.session_state.vectorstore = vectorstore
            st.success(f"{len(fichiers)} fichier(s) indexé(s) en {len(chunks)} fragments.")


# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Zone de question
question = st.chat_input("Posez une question sur vos documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources = None  # pour le mode recherche sémantique, pas de "sources" séparées

    if st.session_state.vectorstore is None:
        reponse = "Veuillez d'abord indexer des documents."
    elif not llm_active:
        # Mode Recherche Sémantique pure
        resultats = rechercher(st.session_state.vectorstore, question)

        reponse = "**Résultats de la recherche sémantique :**\n\n"
        for i, doc in enumerate(resultats, 1):
            source = doc.metadata.get("source", "inconnu")
            page = doc.metadata.get("page", "inconnu")
            reponse += f"**Extrait {i}** (source : *{source}* - page *{page}*)\n\n"
            reponse += f" {doc.page_content}\n\n---\n\n"
    else:
        # Mode RAG complet : Etape 4"
        llm = charger_llm()
        with st.spinner("Génération de la réponse..."):
            reponse_llm, resultats = generer_reponse(st.session_state.vectorstore, llm, question)
        reponse = reponse_llm  # texte affiché dans le chat
        sources = resultats

    # with st.chat_message("assistant"):
    #     st.markdown(reponse)
    # st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
        if sources:
            with st.expander("🔍 Vérifier les sources utilisées"):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "inconnu")
                    st.markdown(f"**Extrait {i}** — *{src}*")
                    st.write(doc.page_content)
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse,
        "sources": sources,  # None si mode recherche sémantique
    })

