import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import Tool
from langchain import SerpAPIWrapper
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import constants as ct
import functions as ft

# 各種設定
st.set_page_config(
    page_title=ct.APP_NAME
)
load_dotenv()

# CSSでの細かいスタイリング
st.markdown(ct.STYLE, unsafe_allow_html=True)

# 初期表示
st.markdown(f"## {ct.APP_NAME}")
st.markdown("**※AIエージェントの利用有無**")

col1, col2 = st.columns([1, 3])
with col1:
    st.session_state.mode = st.selectbox(label="モード", options=["利用する", "利用しない"], label_visibility="collapsed")

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.success("こちらは弊社に関する質問にお答えするチャットボットです。上記の回答モードでAIエージェントの利用有無を選択し、画面下部のチャット欄から自由に質問してください。生成AIロボットが自動でお答えいたします。")
    st.markdown("**【AIエージェントとは】**")
    st.markdown("質問に対して適切と考えられる回答を生成できるまで、生成AIロボット自身に試行錯誤してもらえる機能です。自身の回答に対して評価・改善を繰り返すことで、より優れた回答を生成できます。")
    st.warning("AIエージェントを利用する場合、回答生成により多くの時間がかかる可能性があります。まずはAIエージェントなしで回答を生成し、適切な回答が得られなかった場合にAIエージェントありで回答を生成してみてください。")
    st.caption("※AIエージェントを利用したからといって、必ずしも適切な回答を得られるとは限りません。")

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.feedback_yes_flg = False
    st.session_state.feedback_no_flg = False
    st.session_state.answer_flg = False
    st.session_state.dissatisfied_reason = ""
    st.session_state.feedback_no_reason_send_flg = False

    st.session_state.MAX_ALLOWED_TOKENS = 1000
    st.session_state.total_tokens = 0
    st.session_state.chat_history = []

    # 消費トークン数カウント用のオブジェクト準備
    st.session_state.enc = tiktoken.get_encoding("cl100k_base")
    # コールバックのハンドラ準備
    st.session_state.st_callback = StreamlitCallbackHandler(st.container())
    
    # テーマごとのChainを作成
    service_doc_chain = ft.create_rag_chain(".db_service")
    customer_doc_chain = ft.create_rag_chain(".db_customer")
    company_doc_chain = ft.create_rag_chain(".db_company")
    st.session_state.rag_chain = ft.create_rag_chain(".db_all")

    # Toolで実行される関数の定義
    def run_service_doc_chain(param):
        ai_msg = service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
        return ai_msg["answer"]
    def run_customer_doc_chain(param):
        ai_msg = customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
        return ai_msg["answer"]
    def run_company_doc_chain(param):
        ai_msg = company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
        return ai_msg["answer"]

    # AgentExecutorの作成
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "Web検索Tool",
            func=search.run,
            description="自社サービス「HealthX」に関する質問で、Web検索が必要と判断した場合に使う"
        ),
        Tool(
            func=run_service_doc_chain,
            name="自社サービス「EcoTee」について",
            description="自社サービス「EcoTee」に関する情報を参照したい時に使う"
        ),
        Tool(
            func=run_customer_doc_chain,
            name="顧客とのやり取りについて",
            description="顧客とのやりとりに関する情報を参照したい時に使う"
        ),
        Tool(
            func=run_company_doc_chain,
            name="自社「株式会社EcoTee」について",
            description="自社「株式会社EcoTee」に関する情報を参照したい時に使う"
        )
    ]
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, streaming=True)
    st.session_state.agent_executor = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

input_message = st.chat_input("質問を入力してください。")

if st.session_state.feedback_no_flg and input_message:
    st.session_state.feedback_no_flg = False

for index, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="images/ai_icon.jpg"):
            st.markdown(message["content"])
            # ユーザーフィードバック後のAI側の表示
            if index == len(st.session_state.messages) - 1:
                if st.session_state.feedback_yes_flg:
                    st.caption("ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！")
                    st.session_state.feedback_yes_flg = False
                if st.session_state.feedback_no_flg:
                    st.caption("ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。")
                    st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
                    if st.button("送信"):
                        st.session_state.feedback_no_flg = False
                        st.session_state.feedback_no_reason_send_flg = True
                        st.rerun()
                if st.session_state.feedback_no_reason_send_flg:
                    st.session_state.feedback_no_reason_send_flg = False
                    st.caption("ご回答いただき誠にありがとうございます。")
    else:
        with st.chat_message(message["role"], avatar="images/user_icon.jpeg"):
            st.markdown(message["content"])
            # ユーザーフィードバック後のユーザー側の表示
            if index == len(st.session_state.messages) - 1:
                if st.session_state.feedback_yes_flg:
                    st.caption("ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！")
                    st.session_state.feedback_yes_flg = False
                if st.session_state.feedback_no_flg:
                    st.caption("ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。")
                    st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
                    if st.button("送信"):
                        st.session_state.feedback_no_flg = False
                        st.session_state.feedback_no_reason_send_flg = True
                        st.rerun()
                if st.session_state.feedback_no_reason_send_flg:
                    st.session_state.feedback_no_reason_send_flg = False
                    st.caption("ご回答いただき誠にありがとうございます。")

if input_message:
    # 会話履歴の上限を超えた場合、受け付けない
    input_tokens = len(st.session_state.enc.encode(input_message))
    if input_tokens > st.session_state.MAX_ALLOWED_TOKENS:
        with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
            st.error(f"入力されたテキストの文字数が受付上限値（{st.session_state.MAX_ALLOWED_TOKENS}）を超えています。受付上限値を超えないよう、再度入力してください。")
    else:
        st.session_state.total_tokens += input_tokens

        # ユーザーメッセージの追加と表示
        st.session_state.messages.append({"role": "user", "content": input_message})
        with st.chat_message("user", avatar="images/user_icon.jpeg"):
            st.markdown(input_message)
        
        res_box = st.empty()
        with st.spinner("回答生成中..."):
            # AIエージェントの実行
            result = ft.execute_agent_or_chain(input_message, st.session_state.mode, st.session_state.chat_history)
            st.session_state.messages.append({"role": "assistant", "content": result})
        
        # 古い会話履歴を削除
        response_tokens = len(st.session_state.enc.encode(result))
        st.session_state.total_tokens += response_tokens
        while st.session_state.total_tokens > st.session_state.MAX_ALLOWED_TOKENS:
            removed_message = st.session_state.chat_history.pop(1)
            removed_tokens = len(st.session_state.enc.encode(removed_message.content))
            st.session_state.total_tokens -= removed_tokens

        # AIメッセージの表示
        with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
            st.markdown(result)
            st.session_state.answer_flg = True
            st.caption("この回答はお役に立ちましたか？フィードバックをいただくことで、生成AIの回答の質が向上します。")

# ユーザーフィードバックのボタン表示
if st.session_state.answer_flg:
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("はい"):
            st.session_state.answer_flg = False
            st.session_state.feedback_yes_flg = True
            st.rerun()
    with col2:
        if st.button("いいえ"):
            st.session_state.answer_flg = False
            st.session_state.feedback_no_flg = True
            st.rerun()