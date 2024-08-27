import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from RAG_Flow import (
    load_document,
    chunk_data,
    calculate_embedding_cost,
    create_embeddings,
    clear_history,
    create_conversational_retrieval_chain,
    ask_question,
)

load_dotenv(find_dotenv(), override=True)

# st.image("img.png")
st.subheader("RAG: Langchain + OpenAI + Chromabd + Streamlit 🤖")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    uploaded_files = st.file_uploader(
        "Upload files:", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    chunk_size = st.number_input(
        "Chunk size:", min_value=100, max_value=2048, value=512, on_change=clear_history
    )
    k = st.number_input(
        "k", min_value=1, max_value=20, value=3, on_change=clear_history
    )
    add_data = st.button("Add Data", on_click=clear_history)

    if uploaded_files and add_data:
        all_chunks = []
        with st.spinner("Reading, chunking, and embedding files..."):
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                if data:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    all_chunks.extend(chunks)

        st.write(f"Total chunks: {len(all_chunks)}")

        tokens, embedding_cost = calculate_embedding_cost(all_chunks)
        st.write(f"Embedding cost: ${embedding_cost:.4f}")

        vector_store = create_embeddings(all_chunks)
        st.session_state.vector_store = vector_store
        st.session_state.crc_chain = create_conversational_retrieval_chain(vector_store)

        st.success("Files uploaded, chunked, and embedded successfully.")

q = st.text_input("Ask a question about the content of your files:")
if q:
    if "crc_chain" in st.session_state and st.session_state.crc_chain:
        answer = ask_question(q, st.session_state.crc_chain)
        st.text_area("LLM Answer: ", value=answer)

        st.divider()

        if "history" not in st.session_state:
            st.session_state.history = ""

        value = f"Q: {q} \nA: {answer}"
        st.session_state.history = (
            f'{value} \n {"-" * 100} \n {st.session_state.history}'
        )
        h = st.session_state.history
        st.text_area(label="Chat History", value=h, key="history", height=400)
