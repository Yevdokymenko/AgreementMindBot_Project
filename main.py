import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware # Додано для WordPress
from dotenv import load_dotenv
load_dotenv()


# --- Налаштування ---
INDEX_PATH = "faiss_index"
LLM_MODEL = "gpt-3.5-turbo"

# --- Завантаження бази знань ---
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True) # Додано прапорець
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Будемо шукати 3 найбільш релевантні шматки

# --- Створення промпту (інструкції для ШІ) ---
prompt_template = """
Ти - AgreementMindBot, спеціалізований асистент, який відповідає на питання ТІЛЬКИ на основі змісту безпекових угод України.
Твоя задача - давати чіткі, короткі відповіді.
Завжди дотримуйся цього формату:
1.  Спочатку надай відповідь мовою користувача.
2.  Потім наведи точну цитату з документа англійською мовою, яка підтверджує твою відповідь, у блоці "Original Quote:".
3.  Якщо мова користувача українська, переклади цю цитату українською у блоці "Переклад цитати:".

Якщо питання не стосується безпекових угод, або ти не можеш знайти відповідь у наданих документах, дай таку відповідь:
"На жаль, це питання виходить за межі безпекових угод, які я аналізую. Можливо, вас зацікавлять такі питання:" і запропонуй 3 релевантні питання, які можна поставити.

КОНТЕКСТ ДОКУМЕНТІВ:
{context}

ДЕТАЛІ ЗАПИТУ:
Ім'я користувача: {user_name}
Мова користувача: {language}
Питання: {question}

ТВОЯ ВІДПОВІДЬ:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "user_name", "language", "question"])
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
chain = LLMChain(llm=llm, prompt=PROMPT)

# --- Створення API ---
app = FastAPI()

# Додаємо CORS middleware для дозволу запитів з вашого сайту WordPress
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Для тестування можна "*", для продакшену вкажіть "https://agreementmindbot.win"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_name: str
    language: str

@app.post("/query")
def process_query(request: QueryRequest):
    try:
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        result = chain.invoke({
            "context": context,
            "user_name": request.user_name,
            "language": request.language,
            "question": request.question
        })
        return {"answer": result['text']}
    except Exception as e:
        print("!!! ВИНИКЛА ПОМИЛКА ПІД ЧАС ОБРОБКИ ЗАПИТУ !!!")
        print(f"Тип помилки: {type(e).__name__}")
        print(f"Повідомлення: {e}")
        import traceback
        traceback.print_exc() # Це покаже повний шлях помилки
        return {"error": "Помилка на сервері. Деталі див. у консолі сервера."}

# Команда для запуску: uvicorn main:app --reload