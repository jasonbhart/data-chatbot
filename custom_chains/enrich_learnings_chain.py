"""Chain for enriching an existing summary given new work."""

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from custom_tools.llms import gpt35_text, gpt4_text
from custom_prompts.chains import ENRICH_LEARNINGS_TEMPLATE

ENRICH_LEARNINGS_PROMPT = PromptTemplate.from_template(
    ENRICH_LEARNINGS_TEMPLATE)

enrich_learnings_chain = ENRICH_LEARNINGS_PROMPT | gpt4_text | StrOutputParser()
