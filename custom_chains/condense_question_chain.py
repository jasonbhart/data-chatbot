"""Chain to refine a question to be more concise and standalone."""

from operator import itemgetter

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.output_parsers import PydanticOutputParser

from custom_tools.llms import gpt4
from custom_prompts.chains import CONDENSE_QUESTION_TEMPLATE

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    CONDENSE_QUESTION_TEMPLATE)


class PydanticCondenseQuestionOutputResponse(BaseModel):
    """The response from the agent."""
    assumptions: str = Field(
        description="The assumptions that the agent made to arrive at the question"
    )
    question: str = Field(
        description="The condensed, self-contained question from the stakeholder given a conversation"
    )


condense_question_output_parser = PydanticOutputParser(
    pydantic_object=PydanticCondenseQuestionOutputResponse)

# First load input and chat history into input variables
# This needs to be a RunnableMap because its the first input
loaded_input_variables = RunnableMap(
    {
        "answer": itemgetter("answer"),
        "date": itemgetter("date"),
        "chat_history": itemgetter("chat_history"),
    }
)

standalone_question = {
    "answer": lambda x: x["answer"],
    "date": lambda x: x["date"],
    "chat_history": lambda x: x['chat_history']
} | CONDENSE_QUESTION_PROMPT | gpt4 | condense_question_output_parser


# And now we put it all together!
condense_question_chain = loaded_input_variables | standalone_question
