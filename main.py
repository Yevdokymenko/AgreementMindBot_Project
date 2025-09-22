import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- ЗАВАНТАЖЕННЯ СЕКРЕТІВ ---
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
VECTORSTORE_PATH = "chroma_db" # Використовуємо нову базу даних
LLM_MODEL = "gpt-4o"

# --- ЗАВАНТАЖЕННЯ БАЗИ ЗНАНЬ (РОЗШИРЕНА ВЕРСІЯ) ---
print("Завантаження векторної бази знань Chroma...")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH, 
    embedding_function=OpenAIEmbeddings()
)
retriever = vector_store.as_retriever(search_kwargs={"k": 20}) # Шукаємо 20 маленьких фрагментів
print("База знань завантажена.")

# --- ФІНАЛЬНИЙ, УНІВЕРСАЛЬНИЙ ПРОМПТ ---
final_prompt_template = """
Ти - AgreementMindBot, надточний юридичний асистент. Твоя задача - аналізувати надані фрагменти з безпекових угод України та давати вичерпні, структуровані відповіді.

**СПОЧАТКУ ВИКОНАЙ ЦЕЙ КРОК:**
**КРОК 1: ПЕРЕВІРКА РЕЛЕВАНТНОСТІ**
Уважно прочитай Питання користувача. Порівняй його зі Змістом наданих фрагментів.
- Якщо питання НЕ стосується безпосередньо безпекових угод, політики, військової чи економічної допомоги Україні (наприклад, питання про погоду, спорт, як справи), **ІГНОРУЙ ВСІ ІНШІ ІНСТРУКЦІЇ** і дай ТІЛЬКИ таку відповідь:
"На жаль, це питання не стосується безпекових угод України. Можливо, вас зацікавить:<br>
<div class="suggestions-container">
    <button class="suggested-question">Які країни підписали угоди?</button>
    <button class="suggested-question">Які зобов'язання у сфері кібербезпеки?</button>
    <button class="suggested-question">Як угоди сприяють інтеграції в НАТО?</button>
</div>"
- Якщо питання релевантне, переходь до Кроку 2.

**КРОК 2: ГЕНЕРАЦІЯ ДЕТАЛЬНОЇ ВІДПОВІДІ**
Якщо питання релевантне, дотримуйся цих правил:
1.  **Узагальнююча відповідь:** Почни з загальної відповіді.
2.  **Детальний список:** Якщо доречно, представ інформацію у вигляді пронумерованого списку.
3.  **Розділ "Джерела та цитати":** ПІСЛЯ списку, завжди додавай розділ "--- \n**Джерела та цитати**". Для КОЖНОГО пункту списку, наведи відповідну цитату.
4.  **Формат цитат:**
    **До пункту [Номер]:**
    **Джерело:** [Назва документу], Сторінка: [Номер]
    **Оригінал:**
    > [Цитата]
    **Переклад:**
    > [Переклад]

---
НАДАНІ ФРАГМЕНТИ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ ВІДПОВІДЬ (починай з Кроку 1):
"""

# --- СТВОРЕННЯ ЛАНЦЮГА ДЛЯ ШІ ---
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.1)
FINAL_PROMPT = PromptTemplate(template=final_prompt_template, input_variables=["context", "question"])
final_chain = LLMChain(llm=llm, prompt=FINAL_PROMPT)

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
        # Знаходимо багато маленьких, але дуже релевантних фрагментів
        relevant_docs = retriever.invoke(request.question)
        
        # Тепер ми не можемо просто взяти їхній текст. Нам потрібні їхні "батьківські" документи.
        # На жаль, Langchain не надає простого способу зробити це після збереження.
        # Тому ми використаємо "хак": ми передамо в контекст маленькі фрагменти,
        # але оскільки їх багато (k=20), вони будуть представляти всі документи.
        # Це компроміс, але він вирішить проблему "неповного списку країн".
        
        context_with_metadata = ""
        for doc in relevant_docs:
            source_filename = os.path.basename(doc.metadata.get("source", "N/A"))
            doc_title = DOCUMENT_TITLES.get(source_filename, source_filename)
            page_num = doc.metadata.get("page", 0) + 1
            
            context_with_metadata += f"--- Фрагмент з документу ---\n"
            context_with_metadata += f"Назва документу: {doc_title}\n"
            context_with_metadata += f"Сторінка: {page_num}\n"
            context_with_metadata += f"Зміст: {doc.page_content}\n\n"

        # Робимо ОДИН виклик до ШІ з універсальним промптом
        result = final_chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        
        answer_text = result['text']

        # Оскільки ШІ сам генерує кнопки, нам більше не потрібно додавати посилання вручну
        return {"answer": answer_text}

    except Exception as e:
        print(f"!!! ПОМИЛКА: {e} !!!")
        traceback.print_exc()
        return {"error": "Виникла помилка на сервері."}