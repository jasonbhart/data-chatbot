"""Pydantic agent output parser."""
from __future__ import annotations
import json
import re
from typing import Type, TypeVar, Union, List, Any, Dict

from pydantic import BaseModel, ValidationError, Field, validator

from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.schema import OutputParserException

import custom_tools.general as general
from custom_prompts.tools import PYDANTIC_FORMAT_INSTRUCTIONS

T = TypeVar("T", bound=BaseModel)

JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]

# confrigure and load the agent tools
tools = general.get_toolkit()
tool_names = [tool.name for tool in tools]


class PydanticAgentResponse(BaseModel):
    """The response from the agent."""
    reasoning: str = Field(description="The reasoning behind the command")
    command: str = Field(
        description="The command to execute the next step in the plan")
    command_input: str = Field(
        description="The input required for the command to execute the next step in the plan"
    )

    @validator('command')
    def check_command(cls, field):  # pylint: disable=no-self-argument
        """Validate that the command is valid"""
        if not isinstance(field, str):
            raise ValueError(
                f"Command must be a single string value and be one of {tool_names} or "
                "'Final Answer'"
            )
        if field not in ["Final Answer", tool_names]:
            raise ValueError(
                f"Command {field} is not valid. Must be one of {tool_names} or 'Final Answer'"
            )
        return field

    @validator('command_input')
    def check_command_input(cls, field):  # pylint: disable=no-self-argument
        """Validate that the command input is not empty"""
        if not field:
            raise ValueError("Command input cannot be empty.")
        return field


class PydanticAgentOutputParser(AgentOutputParser):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[T]
    extra_instructions: str = ""

    """The pydantic model to parse."""

    def parse(self, text: str) -> T:
        try:
            text = text.strip(" \n")
            text = re.sub(r"\{\{", "{", text)
            text = re.sub(r"\}\}", "}", text)
            text = "\n" + text
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            if "command" in json_object:
                if json_object["command"] == "Final Answer":
                    if json_object["command_input"] and json_object["command_input"] != "df":
                        response = json_object["command_input"]
                    elif json_object["reasoning"]:
                        response = json_object["reasoning"]
                    else:
                        response = "No response"
                    return AgentFinish(
                        return_values={"output": response},
                        log=text,
                    )
                else:
                    return AgentAction(
                        tool=json_object["command"],
                        tool_input=json_object["command_input"],
                        log=text,
                    )
            else:
                raise OutputParserException(
                    "No Command in output", llm_output=text)

        except (json.JSONDecodeError, ValidationError) as validation_error:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {validation_error}"
            raise OutputParserException(msg, llm_output=text)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        if "description" in reduced_schema:
            del reduced_schema["description"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, indent=2)
        schema_str = schema_str.replace("{", "{{").replace("}", "}}")

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(
            schema=schema_str, extra_instructions=self.extra_instructions)

    @property
    def _type(self) -> str:
        return "pydantic"
