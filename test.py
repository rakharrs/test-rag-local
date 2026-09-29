# test.py
from app import extract

class FakeUploadedFile:
    """Simule l'objet retourné par st.file_uploader pour les tests hors Streamlit."""
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._data = f.read()

    def getvalue(self):
        return self._data


import os

if __name__ == "__main__":
    fichiers = [
        FakeUploadedFile("chiffrement.pdf"),
    ]

    documents = extract(fichiers)

    print(f"Nombre de documents extraits : {len(documents)}")
    for doc in documents:
        print("---")
        print("Source :", doc.metadata.get("source"))
        print("Extrait :", doc.page_content[:200])