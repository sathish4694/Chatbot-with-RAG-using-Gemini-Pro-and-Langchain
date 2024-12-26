import streamlit as st
from main import get_webpage_text, get_text_chunks, get_vector_store, get_conversational_chain

# Set page configuration
st.set_page_config(page_title="RAG Chat Bot", page_icon="📚", layout="centered")


st.markdown("""
    <style>
        .main-container {
            background-color: #f0f8ff; /* Alice blue background */
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
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
    </style>
""", unsafe_allow_html=True)

def user_input(user_question):
    """
    Handles user input and displays the conversation history.

    Args:
        user_question: The question entered by the user.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("**User:**", message.content)
        else:
            st.write("**Reply:**", message.content)

def main():
    """
    Main function to execute the Streamlit application.
    """
    # Display logo and title

    st.title("RAG Q&A Chat Bot")

    # Input URLs
    urls = st.text_area("Enter URLs:").splitlines()
    
    if st.button("Process"):
        with st.spinner("Processing..."):
            text = get_webpage_text(urls)
            text_chunks = get_text_chunks(text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Processing complete!")

    # User input
    user_question = st.text_input("Ask a question:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()