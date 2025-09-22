import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback

# Завантажуємо змінні середовища з файлу .env
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
LLM_MODEL = "gpt-4o" # Використовуємо новішу і розумнішу модель

# --- Завантаження бази знань ---
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Збільшили до 10

# --- Створення основного промпту ---
main_prompt_template = """
Ти - AgreementMindBot, надточний юридичний асистент. Твоя задача - аналізувати надані фрагменти з безпекових угод України та давати вичерпні, структуровані відповіді.

**СУВОРІ ПРАВИЛА, ЯКИХ ТРЕБА НЕУХИЛЬНО ДОТРИМУВАТИСЬ:**

1.  **Узагальнююча відповідь:** Спочатку сформулюй загальну відповідь на питання користувача.
2.  **Детальний список:** Якщо питання про зобов'язання, сфери співпраці, допомогу тощо, ОБОВ'ЯЗКОВО представ основну інформацію у вигляді пронумерованого списку. Кожен пункт списку має бути конкретним і відповідати на частину питання.
3.  **Розділ "Джерела та цитати":** ПІСЛЯ списку, завжди додавай розділ "--- \n**Джерела та цитати**". У цьому розділі, для КОЖНОГО пункту з пронумерованого списку (1, 2, 3...), ти ПОВИНЕН навести відповідну цитату з наданого контексту, що його підтверджує.
4.  **Формат цитат:** Кожна цитата має бути оформлена чітко за таким шаблоном:
    **До пункту [Номер пункту]:**
    **Джерело:** [Назва документу], Сторінка: [Номер сторінки]
    **Оригінал:**
    > [Цитата англійською мовою]
    **Переклад:**
    > [Переклад цитати українською мовою]

---
НАДАНІ ФРАГМЕНТИ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ СТРУКТУРОВАНА ТА ДЕТАЛЬНА ВІДПОВІДЬ З ЦИТАТАМИ:
"""

# --- НОВИЙ Промпт для перевірки релевантності ---
relevance_check_prompt_template = """
Проаналізуй питання користувача та наданий контекст. Відповідай ТІЛЬКИ "так" або "ні".
Чи стосується питання користувача безпосередньо змісту наданого контексту про безпекові угоди України?

Контекст:
{context}

Питання користувача: {question}

Відповідь ("так" або "ні"):
"""

# --- Створення ланцюгів для ШІ ---
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)

MAIN_PROMPT = PromptTemplate(template=main_prompt_template, input_variables=["context", "question"])
main_chain = LLMChain(llm=llm, prompt=MAIN_PROMPT)

RELEVANCE_PROMPT = PromptTemplate(template=relevance_check_prompt_template, input_variables=["context", "question"])
relevance_chain = LLMChain(llm=llm, prompt=RELEVANCE_PROMPT)

# --- Створення API ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        
        # Створюємо простий контекст лише для перевірки релевантності
        simple_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Крок 1: Перевірка релевантності
        relevance_result = relevance_chain.invoke({
            "context": simple_context,
            "question": request.question
        })
        is_relevant = 'так' in relevance_result['text'].lower()

        if not is_relevant:
            # Створюємо HTML-відповідь з кнопками
            off_topic_response = """
            На жаль, це питання не стосується безпекових угод України. Можливо, вас зацікавить:<br>
            <div class="suggestions-container">
                <button class="suggested-question">Які країни підписали угоди?</button>
                <button class="suggested-question">Які зобов'язання у сфері кібербезпеки?</button>
                <button class="suggested-question">Як угоди сприяють інтеграції в НАТО?</button>
            </div>
            """
            return {"answer": off_topic_response}

        # Крок 2: Якщо релевантне, готуємо повний контекст і генеруємо відповідь
        context_with_metadata = ""
        unique_sources_for_links = {}

        for doc in relevant_docs:
            source_filename = os.path.basename(doc.metadata.get("source", "Невідоме джерело"))
            doc_title = DOCUMENT_TITLES.get(source_filename, source_filename)
            page_num = doc.metadata.get("page", 0) + 1
            
            slug = os.path.splitext(source_filename)[0].lower().replace(' ', '-').replace('+', '-')
            url = f"https://agreementmindbot.win/agreements/{slug}/"
            unique_sources_for_links[doc_title] = url
            
            context_with_metadata += f"--- Фрагмент з документу ---\n"
            context_with_metadata += f"Назва документу: {doc_title}\n"
            context_with_metadata += f"Сторінка: {page_num}\n"
            context_with_metadata += f"Зміст: {doc.page_content}\n\n"
        
        main_result = main_chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        
        answer_text = main_result['text']
        
        final_answer = answer_text
        # Додаємо посилання, якщо ШІ згадав документ
        for title, url in unique_sources_for_links.items():
            if title in final_answer:
                final_answer = final_answer.replace(title, f"<a href='{url}' target='_blank'>{title}</a>")

        return {"answer": final_answer}

    except Exception as e:
        # ... (код для помилок залишається той самий)
        return {"error": "Помилка на сервері."}