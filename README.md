# RAG: Langchain + OpenAI + Chroma + Streamlit 🤖

RAG (Retrieve-Answer-Generate) is an interactive application that combines the power of Langchain, OpenAI, Chroma, and Streamlit to provide a sophisticated document analysis and question-answering system. It supports multiple document formats, chunks data for optimised processing, and leverages embeddings for precise information retrieval.

See the RAG application in action:

![RAG Interactive Demo](rag_demo.gif)

## Features

- **Document Support**: Load and process documents in PDF, DOCX, or TXT formats.
- **Data Processing**: Efficiently chunk documents and calculate embedding costs.
- **Interactive UI**: Designed with Streamlit for an intuitive and engaging user experience.
- **Conversational Interface**: Employs an AI-powered conversational retrieval chain, enhancing the system's ability to provide accurate answers based on the document content.
- **Memory Functionality**: Maintains a memory of the chat history, allowing the system to reference previous interactions and provide contextually relevant responses in ongoing sessions.
- **Multi-Document Handling**: Allows simultaneous processing of multiple documents, improving workflow efficiency.
- **Real-Time Feedback**: Provides real-time status updates as documents are processed and embeddings are created, enhancing user engagement.
- **Cost Estimation**: Automatically calculates and displays the cost of embeddings, helping manage API usage and budget.
