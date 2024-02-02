"""Chain to take Python code and an error message and fix the code."""
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from custom_prompts.chains import FIX_CODE_TEMPLATE
from custom_tools.llms import gpt4


class PydanticFixCodeResponse(BaseModel):
    """The response from the agent."""
    reasoning: str = Field(
        description="Reasoning used to fix the code"
    )
    fixed_code: str = Field(
        description="Fixed Python code"
    )


visualize_data_output_parser = PydanticOutputParser(
    pydantic_object=PydanticFixCodeResponse)

FIX_CODE_PROMPT = PromptTemplate.from_template(FIX_CODE_TEMPLATE)
fix_code_chain = FIX_CODE_PROMPT | gpt4 | visualize_data_output_parser
