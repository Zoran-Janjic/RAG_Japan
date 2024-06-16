import streamlit as st
from getLLMresponse import query_rag


def create_ui():
    """
    This function creates the user interface for the JapanLaborAi chatbot using Streamlit.
    """

    st.header("Ask JapanLaborAi")

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            background-color: #333; /* Dark background color */
            color: #fff; /* White text color */
            padding: 10px 0;
            z-index: 1000; /* Ensure it appears above other content */
        }
        .footer a {
            color: #80ccff; /* Light blue for link */
            text-decoration: none; /* Remove underline */
        }
        .footer a:hover {
            color: #66b3ff; /* Slightly lighter blue on hover */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add the footer with author information and "Buy me a coffee" link
    st.markdown(
        """
        <div class="footer">
            Made by <a href="https://github.com/ZoranUTF8" target="_blank">Zoran</a>, 2024, Tokyo-Japan | 
            <a href="https://buymeacoff.ee/yourpagelink" target="_blank">Buy me a coffee</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize the chat messages in the session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages, if any
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Create an empty placeholder for the input field
    input_placeholder = st.empty()

    # Input field for the user to enter their question
    prompt = st.chat_input("Insert your question about Japan Labor Law here...")

    # Process the user's question and display the assistant's response
    if prompt:
        input_placeholder.empty()
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display spinner while waiting for the response
        with st.spinner('Getting your answer...'):
            LLM_response_text = query_rag(prompt)
            st.chat_message("assistant").markdown(LLM_response_text)
            st.session_state.messages.append({"role": "assistant", "content": LLM_response_text})
