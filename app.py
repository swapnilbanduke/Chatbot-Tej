import os
import pdfplumber
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import load_prompt
from langchain.text_splitter import CharacterTextSplitter
from streamlit import session_state as ss
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import uuid
import json
import time
import datetime

# Function to check if a string is a valid JSON
def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

if "firebase_json_key" in os.environ:
    firebase_json_key = os.getenv("firebase_json_key")
else:
    firebase_json_key = st.secrets["firebase_json_key"]

firebase_credentials = json.loads(firebase_json_key)

# Function to initialize connection to Firebase Firestore
@st.cache_resource
def init_connection():
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Attempt to connect to Firebase Firestore
try:
    db = init_connection()
except Exception as e:
    st.write("Failed to connect to Firebase:", e)

# Access Firebase Firestore collection
if 'db' in locals():
    conversations_collection = db.collection('conversations')
else:
    st.write("Unable to access conversations collection. Firebase connection not established.")


# Retrieve OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])

# Streamlit app title and disclaimer
st.title("SwapnilGPT - Swapnil's Resume Bot")
st.image("image/jpg_44-2.jpg", use_column_width=True)
with st.expander("⚠️Disclaimer"):
    st.write("""This bot is a LLM trained on GPT-3.5-turbo model to answer questions about Swapnil's professional background and qualifications. Your responses are recorded in a database for quality assurance and improvement purposes. Please be respectful and avoid asking personal or inappropriate questions.""")

# Define file paths and load initial settings
path = os.path.dirname(__file__)
prompt_template = os.path.join(path, "templates/template.json")
prompt = load_prompt(prompt_template)
faiss_index = os.path.join(path, "faiss_index")
data_source = os.path.join(path, "data/scrapped data.csv")

# Function to store conversation in Firebase
def store_conversation(conversation_id, user_message, bot_message, answered):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered
    }
    conversations_collection.add(data)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS index or create a new one if it doesn't exist
if os.path.exists(faiss_index):
    vectors = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
else:
    # Load data from PDF and CSV sources
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=40
    )
    pdf_loader = PyPDFLoader(pdf_source)
    pdf_data = pdf_loader.load_and_split(text_splitter=text_splitter)
    csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
    csv_data = csv_loader.load()
    data = pdf_data + csv_data

    # Create embeddings for the documents and save the index
    vectors = FAISS.from_documents(data, embeddings)
    vectors.save_local("faiss_index")

# Initialize conversational retrieval chain
retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6, "include_metadata": True, "score_threshold": 0.6})
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo-0125', openai_api_key=openai_api_key), 
                                              retriever=retriever, return_source_documents=True, verbose=True, chain_type="stuff",
                                              max_tokens_limit=4097, combine_docs_chain_kwargs={"prompt": prompt})

# Initialize conversational retrieval chain
retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6, "include_metadata": True, "score_threshold": 0.6})


chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo-0125', openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type="stuff",
    max_tokens_limit=40970,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Function to handle conversational chat
def conversational_chat(query):
    with st.spinner("Thinking..."):
        result = chain({
            "system": "You are a Resume Bot, a comprehensive, interactive resource for exploring Swapnil C. Banduke's background, skills, and expertise. Be polite and provide answers based on the provided context only. Only answer questions relevant to Swapnil and his work experience. Answer questions if they are ONLY regarding Swapnil Banduke. If the question is not relevant to Swapnil, reply that you are a Resume bot. You can make up projects with the skills and projects Swapnil has if the question requests a skill set related to data science, machine learning, database management, or computer science",
            "question": query,
            "chat_history": st.session_state['history']
        })
    
    # Check if the result is a valid JSON
    if is_valid_json(result["answer"]):
        data = json.loads(result["answer"])
    else:
        data = json.loads('{"answered":"false", "response":"Hmm... Something is not right. I\'m experiencing technical difficulties. Try asking your question again or ask another question"}')
    
    answered = data.get("answered", "false")
    response = data.get("response", "")
    questions = data.get("questions", [])

    full_response = "--"

    # Append user query and bot response to chat history
    st.session_state['history'].append((query, response))
    
    # Process the response based on the answer status
    if ('I am tuned to only answer questions' in response) or (response == ""):
        full_response = """"Unfortunately, I can't answer this question. My capabilities are limited to providing information about Swapnil's professional background and qualifications. If you have other inquiries, I recommend reaching out to Swapnil on [LinkedIn](https://www.linkedin.com/in/swapnil-banduke). I can answer questions like:\n - What is Swapnil's educational background?\n - Can you list Swapnil's professional experience?\n - What skills does Swapnil possess?"""
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    else:
        markdown_list = ""
        for item in questions:
            markdown_list += f"- {item}\n"
        full_response = response + "\n\n What else would you like to know about Swapnil? You can ask me: \n" + markdown_list
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    
    return full_response

# Initialize session variables if not already present
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-0125"

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        welcome_message ="Welcome! I'm **Resume Bot**, a virtual assistant designed to provide insights into Swapnil C. Banduke's background and qualifications.\n\nFeel free to inquire about any aspect of Swapnil's profile, such as his educational journey, internships, professional projects, areas of expertise in data science, machine learning, database management, or his future goals.\n\n- His Master's in Business Analytics with a focus on Data Science from UTD\n- His track record in roles at companies like Kirloskar Brothers Limited and EVERSANA\n- His proficiency in programming languages, ML frameworks, data visualization tools, and database management systems\n- His passion for leveraging data to drive business impact and optimize performance\n\nWhat would you like to know first? I'm ready to answer your questions in detail."


        message_placeholder.markdown(welcome_message)

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input and display bot response
if prompt := st.chat_input("Ask me about anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        user_input = prompt
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
