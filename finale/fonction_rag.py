import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

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
def charger_llm():
    return Ollama(model=LLM, temperature=0.1)


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