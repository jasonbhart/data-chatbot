"""Chain for imagining a table schema from a natural language question."""

from operator import itemgetter

from typing import List, Union

from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableMap

from custom_prompts.chains import IMAGINE_TABLES_TEMPLATE
from custom_tools.llms import gpt35

IMAGINE_TABLES_FORMAT_PROMPT = PromptTemplate.from_template(
    IMAGINE_TABLES_TEMPLATE)


class PydanticImagineTablesOutputResponse(BaseModel):
    """The response from the agent."""
    table_schemas_converted_to_create_table_statements: Union[str, List[str]] = Field(
        description="The ideal table schema to service the query represented by CREATE TABLE statements"
    )


imagine_tables_output_parser = PydanticOutputParser(
    pydantic_object=PydanticImagineTablesOutputResponse)

loaded_imagine_tables_variables = RunnableMap(
    {
        "query": itemgetter("query"),
        "requested_dimensions": itemgetter("requested_dimensions"),
        "requested_metrics": itemgetter("requested_metrics"),
        "requested_granularity": itemgetter("requested_granularity"),
    }
)

imagine_tables_chain = (
    loaded_imagine_tables_variables |
    IMAGINE_TABLES_FORMAT_PROMPT |
    gpt35 |
    imagine_tables_output_parser
)
