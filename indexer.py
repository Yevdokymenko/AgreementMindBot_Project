import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Будуємо абсолютний шлях
# Шлях до поточної папки (напр., /opt/render/project/src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# "Виходимо" з неї на один рівень вище (до /opt/render/project/)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Тепер шукаємо папку 'documents' там і присвоюємо її змінній DOCUMENTS_PATH
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "documents")

VECTORSTORE_PATH = "chroma_db"

def create_retriever():
    print("Починаю обробку документів для створення просунутого ретривера...")
    
    # Завантажуємо всі документи
    all_docs = []
    for file in os.listdir(DOCUMENTS_PATH):
        file_path = os.path.join(DOCUMENTS_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue # Пропускаємо файли інших типів
            
            print(f"Завантажую: {file}")
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Помилка при обробці файлу {file}: {e}")

    if not all_docs:
        print("Документи не знайдено. Перевірте папку 'documents'.")
        return

    # "Батьківський" спліттер, який створює великі документи
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # "Дитячий" спліттер, який створює маленькі фрагменти для точного пошуку
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    
    # Створюємо векторне сховище (Chroma) та сховище для документів (InMemoryStore)
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=VECTORSTORE_PATH # Вказуємо папку для збереження
    )
    store = InMemoryStore()

    # Створюємо сам ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Додаю документи до ретривера... Це може зайняти значний час.")
    # Додаємо документи. Цей процес автоматично розіб'є їх на батьківські та дочірні частини.
    retriever.add_documents(all_docs, ids=None)
    
    print(f"База знань успішно створена та збережена у папці: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    create_retriever()