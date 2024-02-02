"""This file contains the LLMs used by the chatbot."""
import os
from pathlib import Path

from dotenv import load_dotenv

import langchain
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2') == 'true'
if LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

PROMPTLAYER_API_KEY = os.getenv('PROMPTLAYER_API_KEY')
if PROMPTLAYER_API_KEY is None:
    ChatModel = ChatOpenAI
else:
    ChatModel = PromptLayerChatOpenAI

CACHE = os.getenv('CACHE') == 'true'

VERBOSE = os.getenv('VERBOSE') == 'true'

if CACHE:
    langchain.llm_cache = InMemoryCache()

# load the LLM models
gpt35 = ChatModel(
    cache=CACHE,
    model="gpt-3.5-turbo-1106",
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=30,
    max_tokens=1000,
    verbose=VERBOSE
)
gpt35_text = ChatModel(
    cache=CACHE,
    model="gpt-3.5-turbo-1106",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=30,
    max_tokens=1000,
    verbose=VERBOSE
)
gpt4 = ChatModel(
    cache=CACHE,
    model="gpt-4-1106-preview",
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=60,
    max_tokens=1000,
    verbose=VERBOSE
)
gpt4_text = ChatModel(
    cache=CACHE,
    model="gpt-4-1106-preview",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=60,
    max_tokens=1000,
    verbose=VERBOSE
)
embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    request_timeout=10,
    openai_api_key=OPENAI_API_KEY
)
