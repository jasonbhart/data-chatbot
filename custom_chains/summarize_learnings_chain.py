"""Chain to summarize learnings from a work log."""

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from custom_tools.llms import gpt4_text
from custom_prompts.chains import SUMMARIZE_LEARNINGS_TEMPLATE

SUMMARIZE_LEARNINGS_PROMPT = PromptTemplate.from_template(
    SUMMARIZE_LEARNINGS_TEMPLATE)

summarize_learnings_chain = SUMMARIZE_LEARNINGS_PROMPT | gpt4_text | StrOutputParser()
