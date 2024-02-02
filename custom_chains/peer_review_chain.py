""""""

from operator import itemgetter

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from custom_prompts.chains import (
    DATA_ANALYST_TEMPLATE,
    DATA_ENGINEER_TEMPLATE,
    DATA_SCIENTIST_TEMPLATE,
    PEER_REVIEW_TEMPLATE,
)
from custom_tools.llms import gpt35_text

data_analyst_chain = PromptTemplate.from_template(DATA_ANALYST_TEMPLATE) | gpt35_text(
    tags=["data_analyst"]) | StrOutputParser()

data_engineer_chain = PromptTemplate.from_template(DATA_ENGINEER_TEMPLATE) | gpt35_text(
    tags=["data_engineer"]) | StrOutputParser()

data_scientist_chain = PromptTemplate.from_template(DATA_SCIENTIST_TEMPLATE) | gpt35_text(
    tags=["data_scientist"]) | StrOutputParser()

peer_review_chain = {
    "data_analyst": data_analyst_chain,
    "data_engineer": data_engineer_chain,
    "data_scientist": data_scientist_chain,
    "query": itemgetter("query"),
    "metrics": itemgetter("metrics"),
    "dimensions": itemgetter("dimensions"),
    "granularity": itemgetter("granularity"),
    "learned_context": itemgetter("learned_context")
} | PromptTemplate.from_template(PEER_REVIEW_TEMPLATE) | gpt35_text | StrOutputParser()
