import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()

def create_rag_chain(db_name):
    """
    会話履歴の記憶機能を持つRAGのChain作成
    Args:
        db_name: データベース名
    """
    if db_name == ".db_service":
        folder_name = "data/service"
    elif db_name == ".db_customer":
        folder_name = "data/customer"
    elif db_name == ".db_company":
        folder_name = "data/company"
    else:
        folder_name = "data"
    files = os.listdir(folder_name)
    docs = []
    for file in files:
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(f"{folder_name}/{file}")
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(f"{folder_name}/{file}")
        else:
            continue
        pages = loader.load()
        docs.extend(pages)
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
        separator="\n",
    )
    splitted_pages = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, streaming=True)

    if os.path.isdir(db_name):
        db = Chroma(persist_directory=db_name, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_pages, embedding=embeddings, persist_directory=db_name)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    question_generator_template = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = """
    あなたは顧客からの質問に対して、分かりやすく丁寧な口調で回答するアシスタントです。
    contextを使用して質問に答えてください。
    また複雑な質問の場合、各項目についてそれぞれ詳細に答えてください。
    答えが分からない場合は、無理に答えようとせず「分からない」という旨を丁寧な表現で答えてください。

    {context}
    """
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def execute_agent_or_chain(input_message, mode, chat_history):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行
    Args:
        input_message: ユーザーメッセージ
        mode: AIエージェントの利用モード
        chat_history: 会話履歴
    """
    if mode == "利用する":
        st_callback = StreamlitCallbackHandler(st.container())
        result = st.session_state.agent_executor.invoke({"input": input_message}, {"callbacks": [st_callback]})
        response = result["output"]
    else:
        result = st.session_state.rag_chain.invoke({"input": input_message, "chat_history": chat_history})
        st.session_state.chat_history.extend([HumanMessage(content=input_message), AIMessage(content=result["answer"])])
        response = result["answer"]
    
    return response