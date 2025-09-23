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
from langchain_community.document_loaders import Docx2txtLoader

load_dotenv()

# --- Словник для перетворення назв файлів на офіційні назви угод ---
DOCUMENT_TITLES = {
    "Dovidka_09_2025.docx": "Зведена довідка по безпекових угодах",
    "2024-02-16-ukraine-sicherheitsvereinbarung-eng-data.pdf": "Угода про співробітництво у сфері безпеки та довгострокову підтримку між Україною та Федеративною Республікою Німеччина",
    "Accord on Support for Ukraine and Cooperation between Ukraine and the Government of Japan.pdf": "Угода про підтримку України та співробітництво між Урядом Японії та Україною",
    "Accordo_Italia-Ucraina_20240224.docx": "Угода про співробітництво у сфері безпеки між Італією та Україною",
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
REFERENCE_FILE_NAME = "Dovidka_09_2025.docx"

# --- НАЛАШТУВАННЯ ---
VECTORSTORE_PATH = "chroma_db"
LLM_MODEL_MAIN = "gpt-4o"
LLM_MODEL_CLASSIFY = "gpt-4o-mini"

# --- ЗАВАНТАЖЕННЯ БАЗИ ЗНАНЬ З УГОДАМИ (КРОК 1) ---
print("Завантаження векторної бази знань Chroma (для угод)...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 50}) 
print("База знань з угодами завантажена.")

# --- ЗАВАНТАЖЕННЯ ФАЙЛУ-ДОВІДКИ В ПАМ'ЯТЬ (КРОК 2) ---
print("Завантаження файлу-довідки...")
# Будуємо абсолютний шлях до папки 'documents' відносно поточного файлу
# Шлях до поточної папки (напр., /opt/render/project/src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# "Виходимо" з неї на один рівень вище (до /opt/render/project/)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Тепер шукаємо папку 'documents' там
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")

reference_loader = Docx2txtLoader(os.path.join(DOCUMENTS_DIR, REFERENCE_FILE_NAME))

reference_text = reference_loader.load()[0].page_content
print("Файл-довідка завантажено.")

# --- ЛАНЦЮГИ ШІ ---
classifier_prompt_template = """Проаналізуй питання користувача. Класифікуй його на один з трьох типів:
1. 'general_listing': Якщо це загальне питання, що вимагає переліку інформації з УСІХ документів (напр., "хто підписав", "які країни", "перелічи всі угоди").
2. 'specific_question': Якщо це конкретне питання про деталі однієї або кількох угод (напр., "що Естонія пообіцяла?", "скільки літаків?", "зобов'язання Франції").
3. 'irrelevant': Якщо питання не стосується угод (напр., "яка погода?", "як справи?").
Відповідай ТІЛЬКИ одним словом: 'general_listing', 'specific_question' або 'irrelevant'.
Питання: "{question}"
Тип:"""
CLASSIFIER_PROMPT = PromptTemplate.from_template(classifier_prompt_template)
classifier_chain = LLMChain(llm=ChatOpenAI(model_name=LLM_MODEL_CLASSIFY, temperature=0), prompt=CLASSIFIER_PROMPT)

# ФІНАЛЬНИЙ ПРОМПТ
main_prompt_template = """
Ти - AgreementMindBot, експерт-аналітик з безпекових угод України.
Тобі надано фрагменти з угод та, що найважливіше, фрагменти зі "Зведеної довідки по безпекових угодах".

**ТВОЯ СТРАТЕГІЯ:**
1.  **ПЕРШ ЗА ВСЕ, ШУКАЙ ВІДПОВІДЬ У "ЗВЕДЕНІЙ ДОВІДЦІ"**. Це твоє головне джерело для структури відповіді.
2.  **ПОТІМ, ВИКОРИСТОВУЙ ТЕКСТИ УГОД ДЛЯ ПОШУКУ КОНКРЕТНИХ ЦИТАТ**, щоб підтвердити кожен пункт, знайдений у довідці.

**ПРАВИЛА ВІДПОВІДІ:**
- Починай з короткого резюме ("**Якщо коротко:**").
- Основну інформацію подавай у вигляді списку під заголовком "**Деталі:**".
- **ЗАБОРОНА:** Ніколи не згадуй "Зведену довідку" або "надані фрагменти".
- **ОБОВ'ЯЗКОВО:** Для кожного пункту списку надай цитату з ОРИГІНАЛЬНОЇ УГОДИ.
- Вказуй повну назву документа та розділ.
- В кінці додай дисклеймер: "Відповідь сформована ШІ...".

**ОСОБЛИВА ІНСТРУКЦІЯ для питання "Хто підписав угоди?":**
Спираючись на "Зведену довідку", склади ПОВНИЙ список усіх країн, які ти там знайдеш. Потім для кожної країни знайди ім'я підписанта в текстах угод.

---
НАДАНІ ДАНІ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ ВІДПОВІДЬ:
"""
MAIN_PROMPT = PromptTemplate(template=main_prompt_template, input_variables=["context", "question"])
main_chain = LLMChain(llm=ChatOpenAI(model_name=LLM_MODEL_MAIN, temperature=0.1), prompt=MAIN_PROMPT)

# --- API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str; user_name: str; language: str

@app.post("/query")
def process_query(request: QueryRequest):
    try:
        # --- КРОК 1: КЛАСИФІКАЦІЯ ПИТАННЯ ---
        classification_result = classifier_chain.invoke({"question": request.question})
        classification = classification_result['text'].strip().lower()

        # Якщо питання нерелевантне, одразу повертаємо відповідь з кнопками
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

        # --- КРОК 2: ВИБІР СТРАТЕГІЇ ТА ФОРМУВАННЯ КОНТЕКСТУ ---
        context_with_metadata = ""
        unique_sources_for_links = {}
        
        if "general_listing" in classification:
            print("Стратегія: Загальний перелік. Використовую довідку + пошук.")
            # Спочатку додаємо в контекст ВЕСЬ текст довідки
            context_with_metadata = f"--- Фрагмент ---\nНазва: Зведена довідка по безпекових угодах\nЗміст: {reference_text}\n\n"
            # Потім додаємо результати звичайного пошуку, щоб знайти додаткові деталі та цитати
            relevant_docs = retriever.invoke(request.question)
        else: # 'specific_question'
            print(f"Стратегія: Конкретне питання. Використовую векторний пошук.")
            relevant_docs = retriever.invoke(request.question)

        # Обробляємо знайдені документи (або для збагачення довідки, або як основне джерело)
        for doc in relevant_docs:
            source_filename = os.path.basename(doc.metadata.get("source", "N/A"))
            doc_title = DOCUMENT_TITLES.get(source_filename, "Невідомий документ")
            page_num = doc.metadata.get("page", 0) + 1
            
            if doc_title != "Невідомий документ" and "довідка" not in doc_title.lower():
                slug = os.path.splitext(source_filename)[0].lower().replace(' ', '-').replace('+', '-')
                url = f"https://agreementmindbot.win/agreements/{slug}/"
                unique_sources_for_links[doc_title] = url
            
            context_with_metadata += f"--- Фрагмент ---\nНазва: {doc_title}\nЗміст: {doc.page_content}\n\n"
        
        # --- КРОК 3: ГЕНЕРАЦІЯ ВІДПОВІДІ ---
        main_result = main_chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        answer_text = main_result['text']
        
        # --- КРОК 4: ДОДАВАННЯ ПОСИЛАНЬ ---
        final_answer = answer_text
        used_titles = {title for title, url in unique_sources_for_links.items() if title in answer_text}
        
        if used_titles:
            final_answer += "\n\n---\n**Пов'язані документи:**"
            for title in sorted(list(used_titles)):
                url = unique_sources_for_links[title]
                final_answer += f"\n* [{title}]({url})"

        return {"answer": final_answer}

    except Exception as e:
        print(f"!!! ПОМИЛКА: {e} !!!")
        traceback.print_exc()
        return {"error": "Виникла помилка на сервері."}