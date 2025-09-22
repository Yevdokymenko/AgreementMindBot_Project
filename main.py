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

# --- Словник для перетворення назв файлів на офіційні назви угод ---
DOCUMENT_TITLES = {
    "2024-02-16-ukraine-sicherheitsvereinbarung-eng-data.pdf": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Федеративною Республікою Німеччина",
    "Accord on Support for Ukraine and Cooperation between Ukraine and the Government of Japan.pdf": "Угода про підтримку України та співробітництво між Урядом Японії та Україною",
    "Accordo_Italia-Ucraina_20240224.pdf": "Угода про співробітництво у сфері безпеки між Італією та Україною",
    "Agreement between Ukraine and the Republic of Latvia on long-term support and security commitments.docx": "Угода між Україною та Латвійською Республікою про довгострокову підтримку та безпекові зобов'язання",
    "Agreement on Long-Term Cooperation and Support between Ukraine and the Republic of Croatia.docx": "Угода про довгострокову співпрацю та підтримку між Україною та Республікою Хорватія",
    "Agreement on Security Cooperation and Long-Term Support between the Kingdom of Belgium and Ukraine.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Королівством Бельгія та Україною",
    "agreement-on-security-cooperation-and-long-time-suppert-no-ukr.pdf": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Королівством Норвегія та Україною",
    "agreement-on-security-cooperation-between-sweden-and-ukraine.pdf": "Угода про співробітництво у сфері безпеки між Швецією та Україною",
    "Agreement on security cooperation and long-term support between Ukraine and Denmark.pdf": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Данією",
    "Agreement on Security Cooperation and Long-term Support between Ukraine and Estonia.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Естонією",
    "Agreement on Security Cooperation and Long-Term Support between Ukraine and Iceland.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Ісландією",
    "Agreement on Security Cooperation and Long-Term Support Between Ukraine and the Czech Republic.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Чеською Республікою",
    "Agreement on security cooperation and long-term support between Ukraine and the Grand-Duchy of Luxembourg.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Великим Герцогством Люксембург",
    "Agreement on security cooperation and long-term support between Ukraine and the Republic of Albania.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Республікою Албанія",
    "Agreement on security cooperation and long-term support between Ukraine and the Republic of Finland.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Фінляндською Республікою",
    "Agreement on Security Cooperation and Long-Term Support Between Ukraine and the Republic of Slovenia.docx": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Республікою Словенія",
    "Agreement on security cooperation between Canada and Ukraine.docx": "Угода про співробітництво у сфері безпеки між Канадою та Україною",
    "Agreement on security cooperation between France and Ukraine (https-:www.elysee.fr:en:emmanuel-macron:2024:02:16:agreement-on-security-cooperation-between-france-and-ukraine).docx": "Угода про співробітництво у сфері безпеки між Францією та Україною",
    "Agreement on Security Cooperation between Spain and Ukraine.docx": "Угода про співробітництво у сфері безпеки між Іспанією та Україною",
    "Agreement on Security Cooperation between Ukraine and Portugal.docx": "Угода про співробітництво у сфері безпеки між Україною та Португалією",
    "Agreement on Security Cooperation Between Ukraine and Romania.docx": "Угода про співробітництво у сфері безпеки між Україною та Румунією",
    "Agreement on Security Cooperation between Ukraine and the Republic of Lithuania.docx": "Угода про співробітництво у сфері безпеки між Україною та Литовською Республікою",
    "Agreement on Security Cooperation between Ukraine and the Republic of Poland.docx": "Угода про співробітництво у сфері безпеки між Україною та Республікою Польща",
    "Agreement on Support for Ukraine and Cooperation between Ukraine and Ireland.docx": "Угода про підтримку України та співробітництво між Україною та Ірландією",
    "Agreement+on+Security+Cooperation+between+the+Netherlands+and+Ukraine.pdf": "Угода про співробітництво у сфері безпеки між Нідерландами та Україною",
    "Bilateral security agreement between Ukraine and the United States of America.docx": "Двостороння безпекова угода між Україною та Сполученими Штатами Америки",
    "Joint Security Commitments between Ukraine and the European Union.docx": "Спільні безпекові зобов'язання між Україною та Європейським Союзом",
    "UK-Ukraine_Agreement_on_Security_Co-operation.pdf": "Угода про співробітництво у сфері безпеки між Україною та Сполученим Королівством Великої Британії і Північної Ірландії"
}

# --- Налаштування ---
INDEX_PATH = "faiss_index"
LLM_MODEL = "gpt-3.5-turbo"

# --- Завантаження бази знань ---
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True) # Додано прапорець
retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Будемо шукати 3 найбільш релевантні шматки

# --- Створення промпту (інструкції для ШІ) ---
prompt_template = """
Ти - AgreementMindBot, експертний асистент, який відповідає на питання ТІЛЬКИ на основі наданих фрагментів з безпекових угод України.
Твоя задача - дати максимально повну та структуровану відповідь, синтезуючи інформацію з УСІХ наданих фрагментів.

ЗАВЖДИ дотримуйся цього формату:
1.  **Відповідь:** Сформулюй комплексну відповідь на питання користувача мовою користувача. Якщо питання передбачає перелік (наприклад, "які зобов'язання"), використовуй марковані списки для чіткості.
2.  **Джерела та цитати:** Після відповіді, для її підтвердження, наведи блок "Джерела та цитати". Для КОЖНОГО ключового твердження у своїй відповіді ти повинен навести цитату, що його підтверджує.

Формат блоку "Джерела та цитати":
---
**Джерело:** [Назва документу], Сторінка: [Номер сторінки, якщо відомий]
**Оригінал:**
> [Цитата з документа англійською мовою]
**Переклад:**
> [Переклад цієї цитати українською]
---
(Повтори цей блок для кожної цитати, яку ти використовуєш, щоб підтвердити свою відповідь)

Якщо питання не стосується безпекових угод або ти не можеш знайти відповідь у наданих документах, дай таку відповідь:
"На жаль, це питання виходить за межі безпекових угод, які я аналізую. Можливо, вас зацікавлять такі питання:" і запропонуй 3 релевантні питання.

НАДАНІ ФРАГМЕНТИ З ДОКУМЕНТІВ (кожен фрагмент містить назву документу, номер сторінки та зміст):
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ ДЕТАЛЬНА ВІДПОВІДЬ:
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
        # retriever.invoke тепер повертає об'єкти Document з метаданими
        relevant_docs = retriever.invoke(request.question)
        
        context_with_metadata = ""
        for doc in relevant_docs:
            # Витягуємо назву файлу з метаданих
            source_filename = os.path.basename(doc.metadata.get("source", "Невідоме джерело"))
            # Беремо красиву назву зі словника, або назву файлу, якщо не знайшли
            doc_title = DOCUMENT_TITLES.get(source_filename, source_filename)
            # Витягуємо номер сторінки (для PDF), якщо він є
            page_num = doc.metadata.get("page", "N/A")
            if page_num != "N/A":
                page_num += 1 # Номерація сторінок починається з 0, робимо її звичною для людей
            
            # Формуємо красивий контекст для передачі в ШІ
            context_with_metadata += f"--- Фрагмент з документу ---\n"
            context_with_metadata += f"Назва документу: {doc_title}\n"
            context_with_metadata += f"Сторінка: {page_num}\n"
            context_with_metadata += f"Зміст: {doc.page_content}\n\n"

        # Викликаємо ланцюг з новим, збагаченим контекстом
        result = chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        
        return {"answer": result['text']}
    except Exception as e:
        print("!!! ВИНИКЛА ПОМИЛКА ПІД ЧАС ОБРОБКИ ЗАПИТУ !!!")
        print(f"Тип помилки: {type(e).__name__}")
        print(f"Повідомлення: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Помилка на сервері. Деталі див. у консолі сервера."}

# Команда для запуску: uvicorn main:app --reload