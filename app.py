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
from transformers.utils import logging as util_log
from langchain_core.utils.uuid import uuid7
import traceback
import logging

load_dotenv()
util_log.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

## Arxiv and wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

ddg_search = DuckDuckGoSearchRun()

# -------------------------
# Custom Tool definition
# -------------------------


@tool
def arxiv_search(query: str) -> str:
    """
    Search ArXiv for research papers and scientific publications.
    Use for academic research, latest papers, technical topics,
    machine learning, physics, mathematics, and computer science.
    """
    logger.info(f"Arxiv search tool called: {query}")
    return arxiv.run(query)


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for stable factual information.

    Use this when the question is about:
    - definitions
    - historical facts
    - concepts
    - famous people
    - scientific explanations

    Do NOT use for latest news or current events.
    """
    logger.info(f"Wikipedia tool called: {query}")

    try:
        result = wiki.run(query)
        return result

    except Exception as e:
        logger.exception("Wikipedia search failed")
        return (
            "Wikipedia search failed temporarily. "
            "Try another source or answer from general knowledge"
        )


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information.

    Use this for:
    - latest events
    - recent technology updates
    - current facts
    - information after knowledge cutoff
    """
    logger.info(f"Web search tool called: {query}")
    try:
        result = ddg_search.run(query)
        return result[:4000]
    except Exception as e:
        logger.exception("Web search failed!")
        return "Web search unavailable"


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
# st.sidebar.title("Settings")

# api_key = os.getenv("GROQ_API_KEY")

## Setting Model API key


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


api_key = get_secret("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found")
    st.stop()

# -------------------------
# System Prompt
# -------------------------

system_prompt = """
You are a helpful search assistant.

Rules:
1. Answer directly if you know.
2. Use Wikipedia for general knowledge.
3. Use web search for current information.
4. Never invent facts.
5. Explain results clearly.
"""

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

tools = [wikipedia_search, web_search]

search_agent = create_agent(model=llm_openai, tools=tools, system_prompt=system_prompt)

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(uuid7())}}

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

    logger.info(f"User query received: {prompt}")
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        logger.info(f"Conversation content so far: {st.session_state.messages}")
        # print(st.session_state.messages)

        try:
            response = search_agent.invoke(
                {"messages": st.session_state.messages},
                config={
                    **st.session_state.config,
                    "callbacks": [st_cb],
                    "tags": ["search-engine-agent"],
                    "metadata": {"environment": "huggingface-space"},
                },
            )

            final_answer = response["messages"][-1].content

            logger.info("Agent response generated successfully")
            logger.info(f"Response length: {len(final_answer)} chars")

            st.session_state.messages.append(AIMessage(content=final_answer))
            st.write(final_answer)

        except Exception as e:
            traceback.print_exc()
            logger.exception("Agent execution failed")
            st.error("Apology! Something went wrong. Search caused an unknown error")
