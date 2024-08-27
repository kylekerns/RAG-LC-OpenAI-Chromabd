import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_document(file):
    import os

    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file)
    else:
        print("Document format is not supported!")
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    vector_store = Chroma.from_documents(chunks, embeddings)

    return vector_store


def create_conversational_retrieval_chain(vector_store):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    system_template = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible for all the options below.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context>
    \nIf you do not know, return I Dont Know.
    """

    user_template = """
    Question: ```{question}```
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Link the ChatGPT LLM
        retriever=retriever,  # Link the vector store based retriever
        memory=memory,  # Link the conversation memory
        chain_type="stuff",  # Specify the chain type
        combine_docs_chain_kwargs={"prompt": qa_prompt},  # Use custom prompt
        verbose=False,  # Set to True to enable verbose logging for debugging
    )

    return crc


def ask_question(q, chain):
    answer = chain.invoke({"question": q})
    return answer["answer"]


def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # https://openai.com/pricing
    return total_tokens, total_tokens / 1000 * 0.00013


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]
