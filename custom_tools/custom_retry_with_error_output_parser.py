from typing import Any, Dict, List, Union

from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate

JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]

NAIVE_COMPLETION_RETRY_WITH_ERROR = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Details: {error}
Please try again:"""

NAIVE_RETRY_WITH_ERROR_PROMPT = PromptTemplate.from_template(
    NAIVE_COMPLETION_RETRY_WITH_ERROR
)


class CustomRetryWithErrorOutputParser(RetryWithErrorOutputParser):
    """Retry with error output parser for the agent."""

    def parse(self, completion: str) -> JsonType:
        """Parse the output."""
        return super().parse_with_prompt(completion, NAIVE_RETRY_WITH_ERROR_PROMPT)
