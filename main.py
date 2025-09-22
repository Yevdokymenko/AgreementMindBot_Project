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
Ти - AgreementMindBot, надточний юридичний асистент. Твоя задача - аналізувати надані фрагменти з безпекових угод України та давати вичерпні, структуровані відповіді.

**СУВОРІ ПРАВИЛА, ЯКИХ ТРЕБА ДОТРИМУВАТИСЬ:**

1.  **Формат відповіді:** Завжди починай з прямої відповіді на питання користувача, узагальнюючи основні пункти.
2.  **Структурований список:** Якщо питання передбачає перелік (наприклад, "які зобов'язання?", "перелічи сфери співпраці"), ОБОВ'ЯЗКОВО представи цю інформацію у вигляді пронумерованого списку. Кожен пункт списку має чітко відповідати на частину питання.
3.  **Підтвердження цитатами:** Після основного тексту відповіді завжди додавай розділ "Джерела та цитати". У цьому розділі, для КОЖНОГО пункту з пронумерованого списку, ти повинен навести відповідну цитату з наданого контексту.
4.  **Деталізація джерела:** Вказуй джерело максимально детально. Використовуй формат:
    `**Джерело:** [Назва документу]`
    Якщо у фрагменті є інформація про розділ або пункт (наприклад, "Part II. Defence and Security", "Article 1. Defence and military cooperation"), ОБОВ'ЯЗКОВО додай її. Наприклад:
    `**Джерело:** Угода про співробітництво... (Розділ II. Defence and Security, Стаття 1)`
5.  **Формат цитат:** Кожна цитата має бути у блоці з оригіналом та перекладом.
    **Оригінал:**
    > [Цитата англійською]
    **Переклад:**
    > [Переклад українською]

**Приклад ідеальної відповіді на питання "Які зобов'язання взяла на себе Естонія?":**

Естонія взяла на себе низку ключових зобов'язань у сфері військової та невійськової підтримки України.

1.  **Щорічна військова підтримка:** Естонія зобов'язалася виділяти щонайменше 0.25% свого ВВП на військову допомогу Україні щорічно.
2.  **Лідерство в IT-коаліції:** Естонія, разом з Люксембургом, очолює IT-коаліцію для розбудови інфраструктури ЗСУ.
3.  **Гуманітарна допомога та реконструкція:** Естонія зосередить свої зусилля на відновленні Житомирської області.
... (і так далі для кожного знайденого зобов'язання)

---
**Джерела та цитати**

**До пункту 1:**
**Джерело:** Угода про співробітництво... (Part II. Defence and Security)
**Оригінал:**
> The Estonian Government has set the target to allocate for military support to Ukraine at least 0.25% of GDP annually from 2024 to 2027.
**Переклад:**
> Уряд Естонії встановив ціль виділяти на військову підтримку України щонайменше 0.25% ВВП щорічно з 2024 по 2027 рік.

**До пункту 2:**
... (і так далі для кожної цитати)

---
НАДАНІ ФРАГМЕНТИ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ СТРУКТУРОВАНА ТА ДЕТАЛЬНА ВІДПОВІДЬ:
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
        
        context_with_metadata = ""
        # Створимо унікальний список використаних джерел для посилань
        unique_sources = {} 

        for doc in relevant_docs:
            source_filename = os.path.basename(doc.metadata.get("source", "Невідоме джерело"))
            doc_title = DOCUMENT_TITLES.get(source_filename, source_filename)
            page_num = doc.metadata.get("page", 0) + 1
            
            # Створюємо URL-дружню назву файлу
            slug = os.path.splitext(source_filename)[0].lower().replace(' ', '-').replace('+', '-')
            unique_sources[doc_title] = f"https://agreementmindbot.win/agreements/{slug}/"

            context_with_metadata += f"--- Фрагмент з документу ---\n"
            context_with_metadata += f"Назва документу: {doc_title}\n"
            context_with_metadata += f"Сторінка: {page_num}\n"
            context_with_metadata += f"Зміст: {doc.page_content}\n\n"

        result = chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        
        answer_text = result['text']
        
        # Додаємо блок з посиланнями в кінець відповіді
        if unique_sources:
            answer_text += "\n\n---\n**Пов'язані документи:**\n"
            for title, url in unique_sources.items():
                answer_text += f"* <a href='{url}' target='_blank'>{title}</a>\n"
        
        return {"answer": answer_text}
    except Exception as e:
        print("!!! ВИНИКЛА ПОМИЛКА ПІД ЧАС ОБРОБКИ ЗАПИТУ !!!")
        print(f"Тип помилки: {type(e).__name__}")
        print(f"Повідомлення: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Помилка на сервері. Деталі див. у консолі сервера."}

# Команда для запуску: uvicorn main:app --reload