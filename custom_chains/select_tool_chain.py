"""The chain for selecting a tool from given context."""

from pydantic import BaseModel, Field

#from langchain.prompts import PromptTemplate

from custom_tools import PydanticAgentOutputParser
#from custom_tools.llms import gpt35
#from custom_prompts.chains import SELECT_TOOL_TEMPLATE

#SELECT_TOOL_TEMPLATE_PROMPT = PromptTemplate.from_template(SELECT_TOOL_TEMPLATE)

class PydanticAgentResponse(BaseModel):
    """The response from the agent."""
    command: str = Field(
        description="The command to run the tool"
    )
    command_input: str = Field(
        description="The input to the tool"
    )

pydantic_agent_parser = PydanticAgentOutputParser(
    pydantic_object=PydanticAgentResponse,
)

#select_tool_chain = SELECT_TOOL_TEMPLATE_PROMPT | gpt35 | pydantic_agent_parser
