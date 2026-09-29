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
LLM = "mistral:latest"
PROMPT_TEMPLATE = """Tu es un assistant qui répond aux questions UNIQUEMENT à partir du contexte fourni ci-dessous.
Règles :
- Si la réponse est introuvable dans le contexte, réponds explicitement "Désolé, je ne trouve pas d'information relative à cette question" 
- Ne fais aucune supposition au-delà de ce qui est écrit dans le contexte.
- Réponds de manière claire et concise, en français.

Contexte :
{context}

Question : 
{question}
"""
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

@st.cache_resource
def charger_embeddings():
    # Modèle recommandé sur le tuto Huggingface - Tsy voatery omena token, fa raha misy token dia faster downloads + higher rate limits
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def charger_llm():
    return Ollama(model=LLM, temperature=0.1)

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

def rechercher(vectorstore, question, k=4):
    """Retourne les k chunks les plus proches sémantiquement de la question."""
    resultats = vectorstore.similarity_search(question, k=k)
    return resultats

def generer_reponse(vectorstore, llm, question, k=4):
    resultats = vectorstore.similarity_search(question, k=k)

    # Construction du contexte : on concatène le contenu des chunks
    contexte = "\n\n---\n\n".join(doc.page_content for doc in resultats)

    # Construction du prompt final
    prompt_final = RAG_PROMPT.format(context=contexte, question=question)

    # 4. Appel au LLM local
    reponse_llm = llm.invoke(prompt_final)

    return reponse_llm, resultats

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

