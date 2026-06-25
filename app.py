import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

os.makedirs("./.cache/files", exist_ok=True)
os.makedirs("./.cache/embeddings", exist_ok=True)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True, 
        memory_key="chat_history"
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. 
            Return any relevant text verbatim. 
            If there is no relevant text, return: ''
            
            [Context]
            {context}
            """
        ),
        ("human", "{question}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant. 
            Answer the user's question using the provided extracted parts of a long document AND the previous chat history.
            
            - If the user asks about the document, rely on the extracted document parts. If the answer is not in the document, say you don't know.
            - If the user asks about previous conversations (e.g., "What did I just ask?"), answer based on the chat history.
            -----
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    
    👈 **First, please enter your OpenAI API Key and upload your file on the sidebar.**
    """
)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if not api_key:
    st.warning("Please provide an OpenAI API Key on the sidebar to use the chatbot.")
else:
    llm_map = ChatOpenAI(
        temperature=0.1, 
        openai_api_key=api_key
    )
    llm_final = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=api_key
    )
    
    map_doc_chain = map_doc_prompt | llm_map

    def map_docs(inputs):
        documents = inputs['documents']
        question = inputs['question']
        results = []
        for doc in documents:
            result = map_doc_chain.invoke({
                "context": doc.page_content,
                "question": question
            }).content
            if result.strip() != "":
                results.append(result)
        return "\n\n".join(results)

    if file:
        retriever = embed_file(file, api_key)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        
        message = st.chat_input("Ask anything about your file...")
        
        if message:
            send_message(message, "human")

            current_history = st.session_state["memory"].load_memory_variables({})["chat_history"]

            map_chain = {
                "documents": retriever,
                "question": RunnablePassthrough()
            } | RunnableLambda(map_docs)

            chain = (
                {
                    "context": map_chain,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: current_history
                }
                | final_prompt
                | llm_final
            )
            
            with st.chat_message("ai"):
                response = chain.invoke(message)
                
                st.session_state["memory"].save_context(
                    {"input": message},
                    {"output": response.content}
                )
    else:
        st.session_state["messages"] = []
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )