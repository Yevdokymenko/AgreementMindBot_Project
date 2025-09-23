import os
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Шлях до поточної папки (напр., /opt/render/project/src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# "Виходимо" з неї на один рівень вище (до /opt/render/project/)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Тепер шукаємо папку 'documents' там
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
# Папка 'documents' знаходиться в тій же директорії
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "chroma_db")

REFERENCE_FILE_NAME = "Dovidka_09_2025.docx" 

def create_vector_store():
    print("Починаю обробку угод...")
    
    agreement_docs = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file == REFERENCE_FILE_NAME:
            continue
        file_path = os.path.join(DOCUMENTS_PATH, file)
        try:
            if file.endswith(".pdf"): loader = PyPDFLoader(file_path)
            elif file.endswith(".docx"): loader = Docx2txtLoader(file_path)
            else: continue
            
            print(f"Індексую угоду: {file}")
            agreement_docs.extend(loader.load())
        except Exception as e:
            print(f"Помилка при обробці файлу {file}: {e}")

    if not agreement_docs:
        print("Угоди не знайдено.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_documents(agreement_docs)
    
    print("Створюю векторні представлення...")
    vectorstore = Chroma.from_documents(
        documents=split_texts, 
        embedding=OpenAIEmbeddings(),
        persist_directory=VECTORSTORE_PATH
    )
    
    print(f"База знань для угод створена у папці: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    if os.path.exists(VECTORSTORE_PATH):
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)
        print(f"Стару базу даних '{VECTORSTORE_PATH}' видалено.")
    create_vector_store()