"""Custom Structured Chat Agent for Data Engineer Assistant."""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import os
import re
import random
from pathlib import Path

import logging
import sys

from dotenv import load_dotenv

import tenacity
from tenacity import *

from langchain.agents import StructuredChatAgent
from langchain.schema import AgentAction, AgentFinish, BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.tools import BaseTool
from langchain.callbacks.base import Callbacks
from custom_tools import general
from custom_tools.general import (
    summarize_learnings,
    enrich_learnings
)
from custom_prompts.agents import (
    HUMAN_MESSAGE_TEMPLATE_V4,
    SYSTEM_TEMPLATE_SUFFIX_V4,
    SYSTEM_TEMPLATE_PREFIX_V4
)

# find the .env file and load it
# this sets OpenAI and other service API keys
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

VERBOSE = os.getenv('VERBOSE') == 'true'

if VERBOSE:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
else:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

logger = logging.getLogger(__name__)


class CustomStructuredChatAgentV4(StructuredChatAgent):
    """Structured Chat Agent."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "\nResult:\n"

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "\nCommand:\n```json\n"

    @property
    def _stop(self) -> List[str]:
        return ["Result:"]

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""

        # The number of previous steps to keep in the scratchpad
        KEEP_PREVIOUS_STEPS = 1
        # Generate the agent scratchpad
        scratchpad = ""
        previous_steps = len(intermediate_steps)
        if previous_steps < KEEP_PREVIOUS_STEPS:
            # If the agent has performed less steps than the number of steps to keep,
            # show all steps
            if "agent_prework" in kwargs:
                # If the agent was fed any preprocessing, add it first
                scratchpad = self.llm_prefix
                scratchpad += kwargs["agent_prework"]
            # Add all previous steps to the scratchpad text
            scratchpad += self._construct_scratchpad(intermediate_steps)
        else:
            # If the agent has performed at least as many steps as steps to keep,
            # show the last KEEP_PREVIOUS_STEPS steps
            if KEEP_PREVIOUS_STEPS == 1:
                # If we only want to show the last step, just show the last step
                # This is a special case because _construct_scratchpad expects a list
                scratchpad = self._construct_scratchpad(
                    [intermediate_steps[-1]])
            else:
                # Otherwise, show the last KEEP_PREVIOUS_STEPS steps
                scratchpad = self._construct_scratchpad(
                    intermediate_steps[-KEEP_PREVIOUS_STEPS])

        # If the agent was fed any preprocessing, add it first
        if "agent_prework" in kwargs:
            work_log = self.llm_prefix
            work_log += kwargs["agent_prework"]
        else:
            work_log = ""
        # Generate the full scratchpad text of work done to summarize
        work_log += self._construct_scratchpad(intermediate_steps)
        work_log = "\n".join(work_log.split("\n")[:-3])
        # Summarize the learnings from entire work log
        if general.learned_context == "None":
            general.learned_context = summarize_learnings(
                work_log=work_log,
                query=kwargs["query"],
                date=kwargs["date"]
            )
        else:
            # If the work log is too long, truncate it
            if len(work_log) > 6000:
                work_log = work_log[-6000:]
            general.learned_context = enrich_learnings(
                work_log=work_log,
                query=kwargs["query"],
                existing_summary=general.learned_context,
                sql_query=general.sql_statement,
                date=kwargs["date"]
            )

        new_inputs = {"learned_context": general.learned_context,
                      "agent_scratchpad": scratchpad, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = f"{self.llm_prefix}"
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)

        def random_sort_learned_context(context: str):
            """Randomly sort the learned context."""
            context = context.split("\n")
            random.shuffle(context)
            return "\n".join(context)

        def shuffle_context():
            """Shuffle the context."""
            def shuffle(retry_state):
                if VERBOSE:
                    print("\nInvalid LLM response. Shuffling context...\n")
                    print(str(retry_state))

                retry_state.kwargs["full_inputs"]["learned_context"] = random_sort_learned_context(
                    retry_state.kwargs["full_inputs"]["learned_context"])
            return shuffle

        @retry(
            reraise=False,
            wait=wait_fixed(1),
            stop=stop_after_attempt(5),
            before_sleep=shuffle_context(),
            after=after_log(logger, logging.ERROR)
        )
        def predict(callbacks: Callbacks, full_inputs: Dict[str, Any]):
            full_output = self.llm_chain.predict(
                callbacks=callbacks, **full_inputs)

            parsed_output = self.output_parser.parse(full_output)

            return parsed_output

        return predict(callbacks=callbacks, full_inputs=full_inputs)

    @classmethod
    def create_agent_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = SYSTEM_TEMPLATE_PREFIX_V4,
        suffix: str = SYSTEM_TEMPLATE_SUFFIX_V4,
        human_message_template: str = HUMAN_MESSAGE_TEMPLATE_V4,
        format_instructions: str = "",
        input_variables: Optional[List[str]] = None,
        memory_prompts: Optional[List[BasePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        formatted_tools = "- "
        formatted_tools += "\n- ".join(tool.name +
                                       ': ' + tool.description for tool in tools)
        formatted_tools += (
            "\n- Final Answer: Useful to signal that you have completed servicing the query. "
            "Input is a string explaining why you believe you have completed servicing the query.\n"
        )
        tool_names = ", ".join(['`' + tool.name + '`' for tool in tools])
        tool_names += ", `Final Answer`"
        format_instructions = re.sub(
            "}", "}}", re.sub("{", "{{", format_instructions))
        system_message_template = "\n".join(
            [prefix, formatted_tools, format_instructions, suffix])
        if "agent_prework" in input_variables:
            input_variables.remove("agent_prework")
        if "date" in input_variables:
            input_variables.remove("date")
        _memory_prompts = memory_prompts or []
        messages = [
            SystemMessagePromptTemplate.from_template(system_message_template),
            *_memory_prompts,
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]
        return ChatPromptTemplate(
            input_variables=input_variables,
            messages=messages
        )
