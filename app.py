import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
    DuckDuckGoSearchResults,
)
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.factory import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from transformers.utils import logging
from langchain_core.utils.uuid import uuid7
import traceback

load_dotenv()
logging.set_verbosity_error()

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

ddg_search = DuckDuckGoSearchRun()

#######


@tool
def arxiv_search(query: str) -> str:
    """
    Search ArXiv for research papers and scientific publications.
    Use for academic research, latest papers, technical topics,
    machine learning, physics, mathematics, and computer science.
    """
    return arxiv.run(query)


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for encyclopedic information.
    Use for historical facts, people, places, concepts, and definitions.
    """
    return wiki.run(query)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information and recent events.
    Use when information may not exist in Wikipedia.
    """
    return ddg_search.run(query)


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")

# api_key = os.getenv("GROQ_API_KEY")

try:
    api_key = st.secrets["OPENAI_API_KEY"]

except Exception:
    api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    st.error("OPENAI_API_KEY not found")
    st.stop()

llm_groq = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    streaming=True,
)

llm_openai = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",  # Or "gpt-4o-mini" depending on your needs
    temperature=0,
    streaming=True,
)

# print(arxiv_search.name)
# print(arxiv_search.args)

# print(wikipedia_search.name)
# print(wikipedia_search.args)

# print(web_search.name)
# print(web_search.args)

tools = [wikipedia_search, web_search]

search_agent = create_agent(model=llm_openai, tools=tools)

config = {"configurable": {"thread_id": str(uuid7())}}

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi, I'm a chatbot who can search the web. How can I help?")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# -------------------------
# User Input
# -------------------------

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        print(st.session_state.messages)

        try:
            response = search_agent.invoke(
                {"messages": st.session_state.messages},
                config={**config, "callbacks": [st_cb]},
            )
            final_answer = response["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=final_answer))
            st.write(final_answer)
        except Exception as e:
            traceback.print_exc()
            print(e)
            st.write("Apology! Search caused an unknown error")
