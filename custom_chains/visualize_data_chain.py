"""The chain to visualize data given a dataframe and a query."""

from operator import itemgetter
import io

import pandas as pd
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableMap

from custom_tools.llms import gpt4
from custom_prompts.chains import PRESCRIBE_VISUALIZATION_TEMPLATE


def df_info_to_string(df: pd.DataFrame):
    """Get the info of a dataframe as a string."""
    buffer = io.StringIO()
    df.info(verbose=True, buf=buffer)
    return buffer.getvalue()


VISUALIZE_DATA_PROMPT = PromptTemplate.from_template(
    PRESCRIBE_VISUALIZATION_TEMPLATE)


class PydanticVisualizeResponse(BaseModel):
    """The response from the agent."""
    visualization: str = Field(
        description="The ideal visualization to service the query given the dataframe"
    )
    code: str = Field(
        description="The code to generate the visualization"
    )


visualize_data_output_parser = PydanticOutputParser(
    pydantic_object=PydanticVisualizeResponse)

visualize_data_chain = RunnableMap({
    "question": itemgetter("question"),
    "dimensions": itemgetter("dimensions"),
    "metrics": itemgetter("metrics"),
    "granularity": itemgetter("granularity"),
    "df_head": lambda x: x['df'].head().to_markdown(),
    "df_info": lambda x: df_info_to_string(x['df']),
}) | VISUALIZE_DATA_PROMPT | gpt4 | visualize_data_output_parser
