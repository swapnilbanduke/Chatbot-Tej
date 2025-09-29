import os
import json
import uuid
import time
import datetime
import streamlit as st

# --- LangChain / OpenAI (modern adapters) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter

# Community loaders/vectorstores (new import paths)
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_valid_json(data: str) -> bool:
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

def read_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit, fallback to env, and strip whitespace."""
    val = ""
    try:
        if name in st.secrets:
            val = st.secrets[name]
        else:
            val = os.getenv(name, default)
    except Exception:
        val = os.getenv(name, default)
    return (val or "").strip()

# ------------------------------------------------------------
# Firebase init
# ------------------------------------------------------------
firebase_json_key = read_secret("firebase_json_key")
firebase_credentials = json.loads(firebase_json_key) if firebase_json_key else None

@st.cache_resource
def init_connection():
    if not firebase_credentials:
        raise RuntimeError("Missing firebase_json_key in secrets.")
    cred = credentials.Certificate(firebase_credentials)
    # Avoid 'already initialized' error on reruns
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = None
try:
    db = init_connection()
    conversations_collection = db.collection("conversations")
except Exception as e:
    st.warning(f"Failed to connect to Firebase: {e}")
    conversations_collection = None

# ------------------------------------------------------------
# OpenAI key handling
# ------------------------------------------------------------
openai_api_key = read_secret("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY missing. Add it to Streamlit secrets.")
    st.stop()

# Make sure any libs reading env get the same key
os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------------------------------------------------
# App UI
# ------------------------------------------------------------
st.title("SwapnilGPT - Swapnil's Resume Bot")
st.image("image/jpg_44-2.jpg", use_column_width=True)

with st.expander("⚠️ Disclaimer"):
    st.write(
        """This bot is an LLM-based assistant that answers questions about \
Swapnil C. Banduke's professional background. Some conversation data may be stored \
for quality and improvement. Please keep it professional."""
    )

# Paths
path = os.path.dirname(__file__)
prompt_template = os.path.join(path, "templates/template.json")  # if you rely on it elsewhere
faiss_index = os.path.join(path, "faiss_index")
data_source = os.path.join(path, "data/csv.csv")
pdf_source = os.path.join(path, "data/resume.pdf")

# ------------------------------------------------------------
# Persist conversation
# ------------------------------------------------------------
def store_conversation(conversation_id, user_message, bot_message, answered):
    if not conversations_collection:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered,
    }
    try:
        conversations_collection.add(data)
    except Exception as e:
        st.info(f"(Note) Could not store conversation: {e}")

# ------------------------------------------------------------
# Embeddings (pass key explicitly)
# ------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# ------------------------------------------------------------
# Load or build FAISS index
# ------------------------------------------------------------
if os.path.exists(faiss_index):
    vectors = FAISS.load_local(
        faiss_index,
        embeddings,
        allow_dangerous_deserialization=True  # you’re already opting into this
    )
else:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=40,
    )

    # Load sources
    pdf_loader = PyPDFLoader(pdf_source)
    pdf_data = pdf_loader.load_and_split(text_splitter=text_splitter)

    csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
    csv_data = csv_loader.load()

    data = pdf_data + csv_data

    # Build index
    vectors = FAISS.from_documents(data, embeddings)
    vectors.save_local(faiss_index)

# ------------------------------------------------------------
# Retriever & LLM
# ------------------------------------------------------------
retriever = vectors.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6, "include_metadata": True, "score_threshold": 0.6},
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",  # keep your current choice
    temperature=0.5,
    api_key=openai_api_key,
)

# If you have a custom prompt JSON you want to use in the combine chain,
# load it here. Keeping your original loader if it’s needed by your template.
from langchain.prompts import load_prompt
prompt = load_prompt(prompt_template)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type="stuff",
    # Avoid the incorrect 40970; let LC handle token budget or set a sane value:
    # max_tokens_limit=4097,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# ------------------------------------------------------------
# Chat function
# ------------------------------------------------------------
def conversational_chat(query):
    with st.spinner("Thinking..."):
        result = chain(
            {
                "question": query,
                "chat_history": st.session_state.get("history", []),
            }
        )

    # Expect the model to return JSON in result["answer"]
    if is_valid_json(result["answer"]):
        data = json.loads(result["answer"])
    else:
        data = json.loads(
            '{"answered":"false","response":"Hmm... Something is not right. '
            'I\'m experiencing technical difficulties. Try again or ask another question."}'
        )

    answered = data.get("answered", "false")
    response = data.get("response", "")
    questions = data.get("questions", [])

    # Record history
    st.session_state["history"].append((query, response))

    # Compose final response
    if ("I am tuned to only answer questions" in response) or (response.strip() == ""):
        full_response = (
            "Unfortunately, I can’t answer this question. I only provide information about "
            "Swapnil’s professional background and qualifications. If you have other inquiries, "
            "reach out to Swapnil on [LinkedIn](https://www.linkedin.com/in/swapnil-banduke).\n\n"
            "I can answer questions like:\n"
            "- What is Swapnil’s educational background?\n"
            "- Can you list Swapnil’s professional experience?\n"
            "- What skills does Swapnil possess?"
        )
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    else:
        markdown_list = "".join(f"- {item}\n" for item in questions)
        full_response = (
            response
            + "\n\nWhat else would you like to know about Swapnil? You can ask me:\n"
            + markdown_list
        )
        store_conversation(st.session_state["uuid"], query, full_response, answered)

    return full_response

# ------------------------------------------------------------
# Session init
# ------------------------------------------------------------
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-0125"

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    with st.chat_message("assistant"):
        welcome_message = (
            "Welcome! I'm **Resume Bot**, here to provide insights into "
            "Swapnil C. Banduke's background and qualifications.\n\n"
            "Ask about education, roles, projects, skills (DS/ML/DB/visualization), "
            "or goals. For example:\n"
            "- His Master's in Business Analytics (Data Science) from UTD\n"
            "- Experience at Kirloskar Brothers Limited and EVERSANA\n"
            "- Proficiency in programming, ML frameworks, visualization tools, and databases\n"
            "- How he leverages data to drive business impact\n\n"
            "What would you like to know first?"
        )
        st.markdown(welcome_message)

if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------------
# Minimal sanity test button (optional)
# ------------------------------------------------------------
with st.expander("🔎 Debug (optional)"):
    if st.button("Run embedding sanity test"):
        try:
            _ = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key).embed_query("hello")
            st.success("Embedding sanity test: OK")
        except Exception as e:
            st.error(f"Embedding sanity test failed: {e}")

# ------------------------------------------------------------
# Chat loop
# ------------------------------------------------------------
# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if user_prompt := st.chat_input("Ask me about anything"):
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        answer = conversational_chat(user_prompt)
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
