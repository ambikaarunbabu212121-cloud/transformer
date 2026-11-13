
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------- CONFIGURATION ----------------

load_dotenv()

st.set_page_config(
    page_title="Maestro AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #240D19;
    color: #FCEEF2;
}

[data-testid="stSidebar"] {
    background: #351321;
    border-right: 1px solid #A76B79;
}

[data-testid="stSidebar"] * {
    color: #F5D5DD !important;
}

.main-title {
    color: #E8B4BE;
    font-size: 43px;
    font-weight: 800;
    letter-spacing: 1px;
}

.subtitle {
    color: #D5A0AE;
    font-size: 15px;
    margin-bottom: 25px;
}

[data-testid="stChatMessage"] {
    background: #3B1929;
    border: 1px solid #8F596A;
    border-radius: 15px;
    padding: 15px;
}

.stChatInput textarea {
    background: #3B1929 !important;
    color: #FFFFFF !important;
}

.stChatInput textarea::placeholder {
    color: #D9AEB9 !important;
}

.stButton button {
    background: #C58B9A;
    color: #240D19;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton button:hover {
    background: #E8B4BE;
    color: #240D19;
}

hr {
    border-color: #805060;
}

h1, h2, h3 {
    color: #E8B4BE !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        "<h2 style='color:#E8B4BE;'>✦ Maestro AI</h2>",
        unsafe_allow_html=True
    )

    st.caption("Your Intelligent AI Assistant")

    st.divider()

    st.subheader("⚙️ Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=8192,
        value=2048,
        step=256
    )

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=100,
        value=40
    )

    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=0.95,
        step=0.05
    )

    st.caption(
        "Top K is a UI setting; the Groq chat "
        "endpoint does not expose it as a standard "
        "sampling parameter."
    )

    st.divider()

    st.markdown("**Model Information**")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Provider: Groq")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- API CLIENT ----------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "Groq API key not found. Add GROQ_API_KEY "
        "to your .env file."
    )
    st.stop()

client = Groq(api_key=api_key)


# ---------------- MAIN INTERFACE ----------------

st.markdown(
    '<div class="main-title">✦ Maestro AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An intelligent assistant powered by Groq and Qwen'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- FRONTEND UPLOADER ----------------

uploaded_file = st.file_uploader(
    "📎 Upload a file",
    type=["txt", "pdf", "csv", "docx"],
    help="Upload a document for future document-chat integration."
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.info(
        "File upload UI is ready. Document reading and "
        "RAG retrieval are not included in this chatbot version."
    )


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

prompt = st.chat_input("Ask Maestro AI anything...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:

            api_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Maestro AI, a helpful, "
                        "accurate and professional AI assistant. "
                        "Give clear and well-structured answers."
                    )
                }
            ]

            api_messages.extend(
                st.session_state.messages
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=True
            )

            full_response = ""

            for chunk in response:

                content = chunk.choices[0].delta.content

                if content:
                    full_response += content
                    response_placeholder.markdown(
                        full_response + "▌"
                    )

            response_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            error_message = f"Maestro AI Error: {str(e)}"

            response_placeholder.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
