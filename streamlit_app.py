# Refer to the Notebook below:
# https://github.com/cohere-ai/notebooks/blob/main/notebooks/Multilingual_Search_with_Cohere_and_Langchain.ipynb
# ! pip install cohere langchain qdrant-client tfds-nightly python-dotenv > /dev/null
# Import modules
import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
#from langchain.document_loaders import TextLoader
#import os
#import random
#import textwrap as tr

class user_message:
    def __init__(self, text, user_name="You"):
        self.name = user_name
        self.container = st.empty()
        self.update(text)

    def update(self, text):
        message = f"""<div style='display:flex;align-items:center;justify-content:flex-end;margin-bottom:10px;'>
                     <div style='background-color:{st.get_option("theme.secondaryBackgroundColor")};border-radius:10px;padding:10px;'>
                     <p style='margin:0;font-weight:bold;'>{self.name}</p>
                     <p style='margin:0;color={st.get_option("theme.textColor")}'>{text}</p>
                     </div>
                     <img src='https://i.imgur.com/qxo27Eu.png' style='width:50px;height:50px;border-radius:50%;margin-left:10px;'>
                     </div>
        """
        self.container.write(message, unsafe_allow_html=True)
        return self


class bot_message:
    def __init__(self, text, bot_name="Assistant"):
        self.name = bot_name
        self.container = st.empty()
        self.update(text)

    def update(self, text):
        message = f"""<div style='display:flex;align-items:center;margin-bottom:10px;'>
                    <img src='https://i.imgur.com/rKTnxVN.png' style='width:50px;height:50px;border-radius:50%;margin-right:10px;'>
                    <div style='background-color:st.get_option("theme.backgroundColor");border: 1px solid {st.get_option("theme.secondaryBackgroundColor")};border-radius:10px;padding:10px;'>
                    <p style='margin:0;font-weight:bold;'>{self.name}</p>
                    <p style='margin:0;color={st.get_option("theme.textColor")}'>{text}</p>
                    </div>
                    </div>
        """
        self.container.write(message, unsafe_allow_html=True)
        return self

class meta_message:
    def __init__(self, text, meta_name="Metadata"):
        self.name = meta_name
        self.container = st.empty()
        self.update(text)

    def update(self, text):
        message = f"""<div style='display:flex;align-items:center;margin-bottom:10px;'>
                    <img src='https://i.imgur.com/rKTnxVN.png' style='width:50px;height:50px;border-radius:50%;margin-right:10px;'>
                    <div style='background-color:st.get_option("theme.backgroundColor");border: 1px solid {st.get_option("theme.secondaryBackgroundColor")};border-radius:10px;padding:10px;'>
                    <p style='margin:0;font-weight:bold;'>{self.name}</p>
                    <p style='margin:0;color={st.get_option("theme.textColor")}'>{text}</p>
                    </div>
                    </div>
        """
        self.container.write(message, unsafe_allow_html=True)
        return self


import streamlit_authenticator as stauth
import yaml

from yaml.loader import SafeLoader
with open('./.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    if username == 'jsmith':
#        st.write(f'Welcome *{name}*')
#        st.title('Application 1')
        st.set_page_config("CECI/REI Multilingual Project Docs Chat Bot 🤖", layout="centered")
        st.title("CTCI/REI Multilingual Project Docs Chat Bot 🤖")

        # Sidebar contents
        with st.sidebar:
            #st.title('🤗💬 CTCI/REI LLM Chat App')
            st.markdown('''
            ![CTCI LOGO](https://reithree.blob.core.windows.net/share/logo.png)
            ## About
            - 採用技術為 Multilingual LLM Model 
            - 主要應用於專案備標文件疑義澄清、專案契約條款解譯、專案會議文件總結、品管安衛環文件衝突分析、專案採購文件差異分析
            ''')
            st.write('Build with ❤️ by [CTCI/REI BMC](https://www.ctci.com/)')


        # API Initiation
        #COHERE_API_KEY = st.secrets.COHERE_API_KEY
        #PINECONE_API_KEY = st.secrets.COHERE_API_KEY
        #PINECONE_API_ENV = st.secrets.COHERE_API_KEY
        COHERE_API_KEY = "lXA8OvRRCsJ8fltEZaMkjYSxUPIkH6w3E6ThHfow"
        PINECONE_API_KEY = "10dc94d7-ba3b-4637-bf32-8056981fc9ed"
        PINECONE_API_ENV = "us-west4-gcp-free"

        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_API_ENV  # next to api key in console
        )
        index_name = "reismchen08733" # put in the name of your pinecone index here

        # create embeddings
        embeddings = CohereEmbeddings(
            model="multilingual-22-12", cohere_api_key=COHERE_API_KEY
        )

        # get existing index
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Create our own prompt template
        prompt_template = """Text: {context}

        Question: {question}

        Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        # Session State Initiation
        prompt = st.session_state.get("prompt", None)

        if prompt is None:
            prompt = [{"role": "system", "content": prompt_template}]

        # If we have a message history, let's display it
        for message in prompt:
            if message["role"] == "user":
                user_message(message["content"])
            elif message["role"] == "assistant":
                bot_message(message["content"], bot_name="REI Multilingual Personal Chat Bot")
            elif message["role"] == "metadata":
                meta_message(message["content"], meta_name="Metadata")

        messages_container = st.container()
        question = st.text_input("", placeholder="Type your message here", label_visibility="collapsed")

        if st.button("Run", type="secondary"):
            prompt.append({"role": "user", "content": question})
            chain_type_kwargs = {"prompt": PROMPT}
            with messages_container:
                user_message(question)
                botmsg  = bot_message("...",  bot_name="Multilingual Project Chat Bot")
                metamsg = meta_message("...", meta_name="MetaData")

            qa = RetrievalQA.from_chain_type(
                llm=Cohere(model="command", temperature=0, cohere_api_key=COHERE_API_KEY),
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True,
            )

            answer = qa({"query": question})
            result = answer["result"].replace("\n", "").replace("Answer:", "")

            mdata = ''
            for t in answer["source_documents"]: mdata = mdata + str(t.metadata) + '\n'

            # Update
            with st.spinner("Loading response .."):
                botmsg.update(result)
                metamsg.update(mdata)

            # Add
            prompt.append({"role": "assistant", "content": result})
            prompt.append({"role": "metadata", "content": mdata})

        st.session_state["prompt"] = prompt
    elif username == 'rbriggs':
        st.write(f'Welcome *{name}*')
        st.title('Application 2')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
