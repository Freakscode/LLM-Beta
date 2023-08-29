import streamlit as st


def main():
    st.set_page_config(page_title='BETA-LLM-V1.0', page_icon=':shark:', layout='wide', initial_sidebar_state='auto')
    
    st.header('BETA-LLM-V1.0')
    st.text_input('Ask your question about the documents here:')
    
    with st.sidebar:
        st.subheader('Your documents')
        st.file_uploader('Upload your PDF\'s here and click on \'Process\':', type=['pdf'])
        st.button('Process')

if __name__ == '__main__':
    main()