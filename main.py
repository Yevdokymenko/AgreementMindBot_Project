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

**ПРИКЛАД ВІДПОВІДІ:**

Норвегія зобов'язалася надати Україні комплексну довгострокову підтримку, яка охоплює військову, цивільну та гуманітарну сфери.

Основні зобов'язання включають:
1.  Надання військової допомоги з акцентом на морську безпеку, ППО та бойову авіацію.
2.  Фінансування через програму Нансена, що передбачає виділення значних коштів на період 2023-2027 років.
3.  Сприяння розвитку оборонної промисловості України та її інтеграції в структури НАТО.
4.  Підтримка реформ, необхідних для майбутнього членства України в ЄС та НАТО.

---
**Джерела та цитати**

**До пункту 1:**
**Джерело:** Угода про співробітництво... між Королівством Норвегія та Україною, Сторінка: 3
**Оригінал:**
> Norway's military assistance to Ukraine is focused on maritime security, integrated air and missile defence, and combat aircraft.
**Переклад:**
> Військова допомога Норвегії Україні зосереджена на морській безпеці, інтегрованій протиповітряній та протиракетній обороні, а також бойовій авіації.

**(і так далі для кожного пункту)**

---
НАДАНІ ФРАГМЕНТИ:
{context}

ПИТАННЯ КОРИСТУВАЧА: {question}

ТВОЯ СТРУКТУРОВАНА ТА ДЕТАЛЬНА ВІДПОВІДЬ З ЦИТАТАМИ:
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
        
        # ВАЖЛИВО: Виклик ШІ залишається тим самим
        result = chain.invoke({
            "context": context_with_metadata,
            "question": request.question
        })
        
        answer_text = result['text']
        
        # Тепер ШІ сам має створити список джерел. Ми лише перетворимо їх на клікабельні посилання.
        # Наприклад, якщо ШІ створив рядок "* Угода про співробітництво... між Королівством Норвегія та Україною"
        # Ми знайдемо його і додамо посилання.
        
        final_answer = answer_text
        for title, url in unique_sources_for_links.items():
            if title in final_answer: # Перевіряємо, чи згадав ШІ цей документ у відповіді
                final_answer = final_answer.replace(title, f"<a href='{url}' target='_blank'>{title}</a>")

        return {"answer": final_answer} # Повертаємо текст, де назви документів вже є посиланнями

    except Exception as e:
        print("!!! ВИНИКЛА ПОМИЛКА ПІД ЧАС ОБРОБКИ ЗАПИТУ !!!")
        print(f"Тип помилки: {type(e).__name__}")
        print(f"Повідомлення: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Помилка на сервері. Деталі див. у консолі сервера."}

# Команда для запуску: uvicorn main:app --reload