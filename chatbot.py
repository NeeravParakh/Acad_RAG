import os
import streamlit as st
from console import *
from tools import *

# Page Configuration
st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title("🤖 AI ChatBot")
st.markdown("""
    <style>
        .chat-container {
            border-radius: 10px;
            padding: 15px;
            background-color: #f4f4f4;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Arial', sans-serif;
        }
        .chat-message {
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #d1e7fd;
            align-self: flex-start;
            text-align: left;
            color: #003366;
            font-weight: bold;
        }
        .ai-message {
            background-color: #d4edda;
            align-self: flex-end;
            text-align: left;
            color: #155724;
            font-weight: bold;
        }
        .latex-output {
            font-size: 16px;
            font-family: 'Courier New', monospace;
            color: #000;
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Chat History
st.markdown("### 🗨️ Chat History")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File Handling
if not os.path.exists("files"):
    os.makedirs("files")

if MAKE_CHROMA:
    st.write("🔄 Making Vectorstores.. Please wait...")
    extract_text_from_folder(FILE_PATH)
    st.success("✅ Vectorstores Created Successfully!")

# Form for User Input
with st.form(key="chat_form"):
    question = st.text_input(
        "💬 Type your message here:",
        placeholder="Ask me anything...",
        key="input_box"
    )
    submit_button = st.form_submit_button(label="📤 Send")

# Process User Input
if submit_button and question:
    response = get_answer(question)  # Get chatbot response
    st.session_state.chat_history.append((question, response))  # Store conversation

# Display Chat History
for user_msg, bot_response in reversed(st.session_state.chat_history):  # Show latest on top
    st.markdown(f"<div class='chat-message user-message'>🧑 <b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
    
    if isinstance(bot_response, list):
        answer = bot_response[0]
        metadata = bot_response[1] if len(bot_response) > 1 else None
    else:
        answer = bot_response
        metadata = None
    
    if "$" in answer or "\\" in answer:  # Check if response contains LaTeX
        st.markdown(f"<div class='chat-message ai-message'>🤖 <b>AI:</b></div>", unsafe_allow_html=True)
        st.latex(answer)  # Properly render LaTeX
    else:
        st.markdown(f"<div class='chat-message ai-message'>{answer}</div>", unsafe_allow_html=True)
    
    if metadata:
        for meta in metadata:
            metadata = meta.metadata
            with st.expander("📖 Source Metadata"):
                st.write(f"**File Name:** {metadata['file_name']}")
                st.write(f"**File Path:** {metadata['file_path']}")
                st.write(f"**Page Number:** {metadata['page']}")
                st.markdown("---")
                st.text_area("Page Content", meta.page_content, height=200)

# Clear Chat History Button
if st.button("🧹 Clear Chat History"):
    st.session_state['chat_history'] = []
    st.rerun()

# Footer Note
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
