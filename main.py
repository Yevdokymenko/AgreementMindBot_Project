import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

# Налаштування логування для кращого дебагінгу
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

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

# --- НАЛАШТУВАННЯ ---
VECTORSTORE_PATH = "chroma_db"
LLM_MODEL_MAIN = "gpt-4o"
LLM_MODEL_MULTI_QUERY = "gpt-4o-mini" # Швидка модель для генерації запитів

# --- ЗАВАНТАЖЕННЯ БАЗИ ЗНАНЬ ---
print("Завантаження векторної бази знань Chroma...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
print("База знань завантажена.")

# --- СТВОРЕННЯ MultiQueryRetriever (КРОК 1) ---
# Промпт для генерації альтернативних запитів
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 20}), # Кожен з 5 запитів знайде по 10 фрагментів
    llm=ChatOpenAI(temperature=0, model_name=LLM_MODEL_MULTI_QUERY),
    prompt=QUERY_PROMPT
)

# --- ПРОМПТ ДЛЯ ГЕНЕРАЦІЇ ВІДПОВІДІ (КРОК 2) ---
main_prompt_template = """
Ти - AgreementMindBot, надточний юридичний асистент. Твоя задача - аналізувати надані фрагменти з безпекових угод України та давати вичерпні, структуровані відповіді.

**ПРАВИЛА ФОРМАТУВАННЯ:**
- Замість "**Узагальнююча відповідь:**" використовуй фразу "**Якщо коротко:**".
- Замість "**Детальний список:**" використовуй фразу "**Деталі:**".
- Використовуй Markdown для форматування: `**жирний**` для заголовків.

**ПРАВИЛА ГЕНЕРАЦІЇ ВІДПОВІДІ:**
1.  **ПОВНОТА:** Якщо питання загальне (наприклад, "Які країни підписали угоди?"), твоя відповідь ПОВИННА базуватися на ВСІХ наданих фрагментах, щоб скласти максимально повний список. Не обмежуйся кількома першими знайденими.
2.  **СТРУКТУРА:** Завжди надавай відповідь у вигляді списку.
3.  **ЦИТАТИ:** Для КОЖНОГО пункту списку ОБОВ'ЯЗКОВО знайди та наведи відповідну цитату з контексту. Якщо ти не можеш знайти пряму цитату для якогось пункту, НЕ ДОДАВАЙ цей пункт до відповіді.
4.  **ДЖЕРЕЛО:** У джерелі вказуй повну назву документа. Спробуй також визначити розділ або статтю з тексту фрагмента і додати її. Наприклад: `Джерело: Угода... між Україною та Францією (Part II. COOPERATION IN THE SECURITY FIELD)`.

---
НАДАНІ ФРАГМЕНТИ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ СТРУКТУРОВАНА ТА ДЕТАЛЬНА ВІДПОВІДЬ:
"""

MAIN_PROMPT = PromptTemplate(template=main_prompt_template, input_variables=["context", "question"])
main_llm = ChatOpenAI(model_name=LLM_MODEL_MAIN, temperature=0.1)
main_chain = LLMChain(llm=main_llm, prompt=MAIN_PROMPT)

# --- СТВОРЕННЯ API ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str; user_name: str; language: str

@app.post("/query")
def process_query(request: QueryRequest):
    try:
        # --- КРОК 1: КЛАСИФІКАЦІЯ ПИТАННЯ ---
        classification_result = classifier_chain.invoke({"question": request.question})
        classification = classification_result['text'].strip().lower()

        if "irrelevant" in classification:
            off_topic_response = """
            На жаль, це питання не стосується безпекових угод України. Можливо, вас зацікавить:<br>
            <div class="suggestions-container">
                <button class="suggested-question">Які країни підписали угоди?</button>
                <button class="suggested-question">Які зобов'язання у сфері кібербезпеки?</button>
                <button class="suggested-question">Як угоди сприяють інтеграції в НАТО?</button>
            </div>
            """
            return {"answer": off_topic_response}

        # --- КРОК 2: ЯКЩО РЕЛЕВАНТНЕ, ВИКОРИСТОВУЄМО MultiQueryRetriever для пошуку документів ---
        print(f"Отримано релевантне питання: {request.question}")
        relevant_docs = multi_query_retriever.invoke(request.question)
        print(f"Знайдено {len(relevant_docs)} релевантних фрагментів через MultiQueryRetriever.")

        # --- КРОК 3: ФОРМУЄМО КОНТЕКСТ І ГЕНЕРУЄМО ВІДПОВІДЬ ---
        context_with_metadata = ""
        unique_sources_for_links = {}

        for doc in relevant_docs:
            source_filename = os.path.basename(doc.metadata.get("source", "N/A"))
            doc_title = DOCUMENT_TITLES.get(source_filename, source_filename)
            page_num = doc.metadata.get("page", 0) + 1
            
            slug = os.path.splitext(source_filename)[0].lower().replace(' ', '-').replace('+', '-')
            url = f"https://agreementmindbot.win/agreements/{slug}/"
            if doc_title: # Додаємо тільки якщо назва відома
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
        for title, url in unique_sources_for_links.items():
            if title in final_answer:
                final_answer = final_answer.replace(title, f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>")

        return {"answer": final_answer}

    except Exception as e:
        print(f"!!! ПОМИЛКА: {e} !!!")
        traceback.print_exc()
        return {"error": "Виникла помилка на сервері."}