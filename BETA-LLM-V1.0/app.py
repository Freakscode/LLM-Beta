import streamlit as st
import accelerate
from transformers import pipeline
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()    
    return text


def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=400,
                                          chunk_overlap=250,
                                          length_function=len
                                          )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks


def VectorStore(chunks):
    model = 'hkunlp/instructor-large'
    embeddings = HuggingFaceInstructEmbeddings(model_name=model)
    VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return VectorStore
    

def get_conversation(vectorstore):
    model = 'bigscience/bloom'
    llm = HuggingFaceHub(repo_id=model, model_kwargs={'temperature':0.2,'max_length':500, 'num_beams': 5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='BETA-LLM-V1.0', page_icon=':shark:', layout='wide', initial_sidebar_state='auto')
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header('BETA-LLM-V1.0')
    user_question = st.text_input('Ask your question about the documents here:')
    if user_question:
        handle_userinput(user_question)
      
    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Upload your PDF\'s here and click on \'Process\':', type=['pdf'], accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner('Processing...'):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                chunks = get_chunks(raw_text)
                
                # Create vector store
                vector_store = VectorStore(chunks)
                
                # Assign vector store and create conversation chain
                st.session_state.vector_store = vector_store
                st.session_state.conversation = get_conversation(vector_store)

if __name__ == '__main__':
    main()

