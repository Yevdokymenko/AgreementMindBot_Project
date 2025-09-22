import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


DOCUMENTS_PATH = "documents"
INDEX_PATH = "faiss_index"

def create_vector_store():
    print("Починаю обробку документів...")
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        file_path = os.path.join(DOCUMENTS_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Обробляю PDF: {file}")
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                print(f"Обробляю DOCX: {file}")
        except Exception as e:
            print(f"Помилка при обробці файлу {file}: {e}")

    if not documents:
        print("Документи не знайдено. Перевірте папку 'documents'.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_texts = text_splitter.split_documents(documents)
    
    print("Створюю векторні представлення (embeddings)... Це може зайняти час.")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_texts, embeddings)
    
    print(f"Зберігаю індекс у папку: {INDEX_PATH}")
    vector_store.save_local(INDEX_PATH)
    print("Готово! База знань створена.")

if __name__ == "__main__":
    create_vector_store()