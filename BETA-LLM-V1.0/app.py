import streamlit as st
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
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len
                                          )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks


def VectorStore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return VectorStore
    

def get_conversation(vectorstore):
    model = 'meta-llama/Llama-2-7b-chat-hf'
    llm = HuggingFaceHub(repo_id=model, model_kwargs={'temperature':0,'max_length': 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation


def main():
    load_dotenv()
    st.set_page_config(page_title='BETA-LLM-V1.0', page_icon=':shark:', layout='wide', initial_sidebar_state='auto')
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    st.header('BETA-LLM-V1.0')
    st.text_input('Ask your question about the documents here:')
    
    st.write(user_template.replace("{{MSG}}","Hola bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hola usuario"), unsafe_allow_html=True)    
    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Upload your PDF\'s here and click on \'Process\':', type=['pdf'], accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner('Processing...'):
                #Get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #Get text chunks
                chunks = get_chunks(raw_text)
                
                #Create vector store
                vector_store = VectorStore(chunks)
                
                #Conversation chain
                st.session_state.conversation = get_conversation(vector_store)
    
                
                
if __name__ == '__main__':
    main()