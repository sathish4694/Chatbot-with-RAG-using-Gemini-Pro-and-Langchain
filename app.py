import streamlit as st
from main import get_webpage_text, get_text_chunks, get_vector_store, get_conversational_chain

# Set page configuration
st.set_page_config(page_title="RAG Chat Bot", page_icon="🤖", layout="wide")

# Inject custom styles
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }

        .stButton>button {
            background-color: #00796b; /* Teal background */
            color: white;
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #004d40; /* Darker teal */
            transform: scale(1.05);
        }
        .stTextInput>div>input, .stTextArea>div>textarea {
            border: 2px solid #00796b; /* Teal border */
            border-radius: 8px;
            padding: 14px;
            font-size: 16px;
            background-color: #e0f2f1; /* Light teal background */
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stTextInput>div>input:focus, .stTextArea>div>textarea:focus {
            border-color: #004d40; /* Darker teal border on focus */
            box-shadow: 0 0 8px rgba(0, 77, 64, 0.5);
        }
        .response-box {
            margin-top: 20px;
            padding: 20px;
            background-color: #f1f8e9; /* Light green background */
            border-radius: 8px;
            border: 1px solid #c5e1a5; /* Green border */
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None

def validate_urls(urls):
    """Validates the provided URLs."""
    import re
    valid_urls = [url for url in urls if re.match(r'^https?://', url)]
    if len(valid_urls) < len(urls):
        st.warning("Some URLs were invalid and skipped.")
    return valid_urls

def main():
    """Main function to run the Streamlit application."""
    st.title("🤖 RAG Q&A Chat Bot")
    st.subheader("Interactive chatbot powered by Retrieval-Augmented Generation (RAG).")
    st.markdown(
        """
        Enter URLs below. The bot will analyze the content and be ready to answer your questions!
        """,
        unsafe_allow_html=True,
    )

    # Input URLs
    urls = st.text_area(
        "Enter URLs:",
        placeholder="Paste one URL per line here...",
        height=70,
    ).splitlines()
    urls = validate_urls(urls)

    if st.button("Process"):
        with st.spinner("Analyzing content..."):
            try:
                text = get_webpage_text(urls)
                text_chunks = get_text_chunks(text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Processed! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing URLs: {e}")

    # User question input
    st.markdown("### Ask Your Question")
    user_question = st.text_input(
        "Type your question below:",
        placeholder="What would you like to know?",
    )

    if user_question and st.session_state.conversation:
        try:
            response = st.session_state.conversation({"question": user_question})
            bot_reply = response["chat_history"][-1].content
            st.markdown(
                f"<div class='response-box'><strong>Bot's Answer:</strong><br>{bot_reply}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error during conversation: {e}")

if __name__ == "__main__":
    main()
