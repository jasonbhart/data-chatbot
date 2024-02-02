"""Custom chain prompts for the agents."""

from custom_prompts.agents import (
    PREMISE,
    GENERAL_CONTEXT,
    BUSINESS_CONTEXT
)

QUESTION_TEMPLATE = (
    PREMISE +
    "Before you begin processing the stakeholder query below, you may ask one "
    "question to clarify the request. Only ask a question if it is necessary to "
    "understand the request, stakeholder intent, or related context. "
    "Do not ask questions for which an answer can be reasonably assumed or inferred from the query. "
    "The stakeholder will not know technical details about how to service the query nor details "
    "about the data including how the data was generated, stored, processed. Only ask about the "
    "specifications of what is being requested and how it is to be delivered. "
    "Assume all data exists and is accessible. Assume you will figure out how to retrieve "
    "the data later. Assume the query is asking for aggregates unless specified. "
    "Assume metrics are already calculated in the data you will access. "
    "Be sure to ask questions about any unknowns needed to service the query or which cannot be inferred. "
    "Make sure you understand or can infer the granularity needed to "
    "service the question. Do not include measures or dimensions which are not asked for. "
    "If no further clarification is needed, please output 'no more questions' "
    "as the question. Now, take a deep breath and work on this step by step."
    "\n---\n"
    "Context:\n" + GENERAL_CONTEXT + BUSINESS_CONTEXT +
    "\n---\n"
    "Stakeholder Query: {question}\n\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "question": $str_clarification_question_for_stakeholder,\n  "dimensions": $list_req_dimensions,\n  '
    '"metrics": $list_req_measures,\n  "granularity": $str_granularity\n}}'
    "\n```\n\n"
    "Output: \n")

CONDENSE_QUESTION_TEMPLATE = (
    "Given a conversation between a human and an assistant, create a verbose self-contained "
    "question that accurately represents the human's curent intent and which includes all "
    "relevant context. The question should be standalone and not require any additional "
    "information or access to the conversation history in order to answer it. "
    "Use the conversation history and context to infer any ambiguity. Pay attention to the "
    "clues about the human's most recent intent as it might shift during the conversation. "
    "The question should be phrased in clear and concise English. "
    "Now, take a deep breath and work on this step by step.\n"
    "\n---\n"
    "Context:\n" + GENERAL_CONTEXT + BUSINESS_CONTEXT +
    "\n---\n"
    "Chat History:"
    "{chat_history}"
    "Human: {answer}\n\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "assumptions": $str_concise_reflection_of_assumptions_made\n  "question": $str_standalone_question\n}}'
    "\n```\n\n"
    "Output: \n"
)

PRESCRIBE_OUTPUT_TEMPLATE = (
    "You are a data visualization and business intelligence expert. "
    "Given the following query, prescribe the needed dimensions, measures, and "
    "granularity needed for data storytelling in service of the query below. "
    "Reflect on the stakeholder's implied intent and probable use case."
    "Just answer the question, do not provide any additional information or "
    "explanation. Now, take a deep breath and work on this step by step."
    "\n---\n"
    "Context:\n" + GENERAL_CONTEXT + BUSINESS_CONTEXT +
    "\n---\n"
    "The $granularity must be one of the following: "
    "\n- daily\n- weekly\n- monthly\n- quarterly\n- yearly\n- cohort"
    "\n"
    "Query: {question}\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "dimensions": $list_of_dimensions,\n  "metrics": $list_of_measures,\n  '
    '"granularity": $granularity\n}}'
    "\n```\n\n"
    "Output: \n"
)

PRESCRIBE_VISUALIZATION_PREFIX = (
    "As an expert data storyteller, your task is to create a visualization for the "
    "data in the Pandas dataframe `df` that effectively answers the user's query. "
    "Choose the most suitable visualization and axes based on the dataframe metadata "
    "and user query. Write Python 3.x code using seaborn and pyplot to generate the "
    "visualization, following these guidelines:\n"
    "- Use the dataframe `df` as-is, without modifying or recreating it.\n"
    "- Use `month`, `week`, `day`, and `year` columns as-is. Do not convert them to date or datetime.\n"
    "- Always sort the dataframe by day, week, month, year, or date, from oldest to newest.\n"
    "- Columns shown in the results of df.info() are is the only available data for use.\n"
    "- Ensure your code is valid, self-contained, and follows proper Python indentation "
    "and syntax.\n"
    "- Choose a chart type that accurately represents the type, scale, and distribution "
    "of the data.\n"
    "- Order categorical axes by relevance and date-based axes by time, from left to "
    "right.\n"
    "- Set the origin of numeric-based Y axes to 0.\n"
    "- Fit the axis of the graph to the data so that it fills the graph.\n"
    "- Include labels, titles, and legends for clarity.\n"
    "- Use seaborn style `whitegrid` and function `despine()` if appropriate.\n"
    "- Aggregate the data as needed to answer the query using only the dimensions and "
    "metrics listed below.\n"
    "- When using partial dates, such as month values, include the year context.\n"
    "- Only include the top 5 values of categorical axes in the legend.\n"
    "- Use commas when displaying large numbers.\n"
    "- Label data points of interest, like highs and lows, with their values.\n"
    "- Ensure that the visualization is easy to read and understand.\n"
    "\n"
    "Now, take a deep breath and work on this step by step.\n"
)

PRESCRIBE_VISUALIZATION_TEMPLATE = (
    PRESCRIBE_VISUALIZATION_PREFIX +
    "The $visualization_type must be one of the following: "
    "\n- figure\n- list\n- table\n- pie_chart\n- bar_chart\n- line_chart\n- scatter_plot"
    "\n- box_plot\n- density_plot\n- histogram\n- heatmap\n- treemap\n- sankey_diagram"
    "\n\n"
    "User Query: {question}\n"
    "Dimensions: {dimensions}\n"
    "Metrics: {metrics}\n"
    "Desired Granularity: {granularity}\n"
    "Results of df.head():\n{df_head}\n"
    "Results of df.info():\n{df_info}\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "visualization": $visualization_type, \n  "code": $python_code\n}}'
    "\n```\n\n"
    "Output: \n"
)

FIX_CODE_TEMPLATE = (
    "As an expert Python programmer, your task is to fix erroneous Python 3.x code"
    "which was generated in service of a request. Given the original stakeholder "
    "request, the resulting code, and the AST error from running the code, rewrite the "
    "code to fix the error. Ensure that the fixed code is valid, self-contained, and "
    "generates the desired visualization without relying on any external variables. "
    "Do not change the code in any other way or provide any additional information or "
    "explanation. Now, take a deep breath and work on this step by step.\n"
    "\n"
    "Stakeholder Request:\n"
    "\n###\nErroneous Code:\n{code}\n###\n"
    "Error: {error}\n\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "reasoning": $str_fix_explanation, "fixed_code": $str_python_code\n}}'
    "\n```\n\n"
    "Output: \n"
)

BASE_LEARNINGS_TEMPLATE = (
    "Include:\n"
    "\n"
    "* Relevant learnings and proven hypotheses\n"
    "* Discovered or inferred business context\n"
    "* Salient insights and conclusions\n"
    "* Constructive feedback for future work\n"
    "* Your reflections on the work done so far\n"
    "* Important counterfactual details\n"
    "* Principal tables and fields along with the reason\n"
    "* Derived or inferred data models with relationships\n"
    "* Associations, assertions, and concepts you observed\n"
    "* Other key information that would be useful to the next person to work on the request\n"
    "* The current state of the dataframe `df` (is there data in it?, what data is in "
    "it?, does the data require further processing?, etc.)\n"
    "* If the request has been satisfied and why\n"
    "\n"
    "Do not make up information. Don't include steps which are "
    "unnecessary or go beyond the minimum work needed to staisfy the request. "
    "Do not mention anything about visualizing the data. Use only bullets and do not "
    "include headings or titles. Do not preserve the log's structure or sequencing; "
    "only include a summary of the log. Only use proper nouns when referencing tables "
    "and fields. Do not include items which are not relevant to servicing the request "
    "Now, take a deep breath and work on this step by step.\n"
    "\n"
    "Request: {query}\n"
    "\n---\n"
    "Work Log:\n"
    "{work_log}"
    "\n---\n"

)

SUMMARIZE_LEARNINGS_TEMPLATE = (
    "As an expert data analyst, your task is to efficiently summarize the relevant "
    "learnings, insights, associations, assertions, and concepts discovered "
    "while servicing a stakeholder request. Write a succinct and concise "
    "bulleted summary of the work described in a episodic log.\n" +
    BASE_LEARNINGS_TEMPLATE +
    "\n"
    "Output format:\n"
    "- Bullet 1 (learnings, insights, conclusions, reflections, key information, etc.)\n"
    "- Bullet 2 (learnings, insights, conclusions, reflections, key information, etc.)\n"
    "- Bullet n (if any)\n"
    "\n"
    "Output:\n"
    "- The current date is: {date}\n"
)

ENRICH_LEARNINGS_TEMPLATE = (
    "As an expert data analyst, your task is to contribute to and improve an existing "
    "concise summary of learnings, insights, associations, assertions, and concepts "
    "discovered while servicing a stakeholder request. "
    "Add to and revise this concise, bulleted summary of the work described in a "
    "episodic log. Do not remove items from the existing summary. Revise items to "
    "to reflect current understandings given the work in the log. Keep in mind the work "
    "log may have been truncated due to length and thus may not include all prior work.\n"
    + BASE_LEARNINGS_TEMPLATE +
    "Existing Summary:\n"
    "{learned_context}"
    "\n---\n"
    "\n"
    "Output format:\n"
    "- Bullet 1 (learnings, insights, conclusions, reflections, key information, etc.)\n"
    "- Bullet 2 (learnings, insights, conclusions, reflections, key information, etc.)\n"
    "- Bullet n (if any)\n"
    "\n"
    "Output:\n"
    "- The current date is: {date}\n"
    "{sql_query}"
)

SCHEMA_SEARCH_TEMPLATE = """As an expert database architect, your task is to design an ideal
set of BigQuery GoogleSQl-dialect table schemas and write the corresponding ANSI-compliant
SQL `CREATE TABLE` (ddl) statements you imagine would be needed to service a given stakeholder
request. Assume that the data warehouse uses fully-normalized star-schema modeled tables,
with fact and dimension tables, and that multiple tables may be needed to service the request.
Now, take a deep breath and work on this step by step.

Request: {query}

Output format:
```json
{{
  "table_schemas_converted_to_create_table_statements": $str_bigquery_ddl_create_table_statements
}}
```
Output:
"""

IMAGINE_TABLES_TEMPLATE = """As an expert database architect, your task is to design an ideal
set of BigQuery GoogleSQl-dialect table schemas and write the corresponding ANSI-compliant
SQL `CREATE TABLE` (ddl) statements you imagine would be needed to service a given stakeholder
request. Assume that the data warehouse uses fully-normalized star-schema modeled tables,
with fact and dimension tables, and that multiple tables may be needed to service the request.
Now, take a deep breath and work on this step by step.

Request: {query}
Requested dimensions: {requested_dimensions}
Requested metrics: {requested_metrics}
Requested granularity: {requested_granularity}

Output format:
```json
{{
  "table_schemas_converted_to_create_table_statements": $str_bigquery_ddl_create_table_statements
}}
```
Output:
"""

DATA_ANALYST_TEMPLATE = """
As an expert data analyst, you are responsible for servicing stakeholder requests for analysis.
Given the following stakeholder request and learned context of work done so far, write a bulleted
plan of the work left to be done to service the request. Include your reasoning with each bullet.
Let's think step by step. Don't suggest work which is not explicitly asked for in the request.
Don't suggest visualization. Don't suggest exporting the data from the dataframe. Now, take a deep
breath and work on this step by step.

Stakeholder request: {query}
Required Metrics: {metrics}
Required Data Dimensions: {dimensions}
Required Data Granularity: {granularity}
---
Learned context:
{learned_context}
---
Results of last work step:
{agent_scratchpad}
---
List of work left to be done:
"""

DATA_ENGINEER_TEMPLATE = """
As an expert data engineer, you are responsible for servicing stakeholder requests for data.
Given the following stakeholder request and learned context of work done so far, write a bulleted
plan of the work left to be done to service the request. Include your reasoning with each bullet.
Let's think step by step.  Don't suggest work which is not explicitly asked for in the request.
Don't suggest visualization. Don't suggest exporting the data from the dataframe. Now, take a
deep breath and work on this step by step.

Stakeholder request: {query}
Required Metrics: {metrics}
Required Data Dimensions: {dimensions}
Required Data Granularity: {granularity}
---
Learned context:
{learned_context}
---
Results of last work step:
{agent_scratchpad}
---
List of work left to be done:
"""

DATA_SCIENTIST_TEMPLATE = """
As an expert data scientist, you are responsible for servicing stakeholder requests for advanced analytics.
Given the following stakeholder request and learned context of work done so far, write a bulleted
plan of the work left to be done to service the request. Include your reasoning with each bullet.
Let's think step by step. Don't suggest work which is not explicitly asked for in the request.
Don't suggest visualization. Don't suggest exporting the data from the dataframe. Now, take a deep breath
and work on this step by step.

Stakeholder request: {query}
Required Metrics: {metrics}
Required Data Dimensions: {dimensions}
Required Data Granularity: {granularity}
---
Learned context:
{learned_context}
---
Results of last work step:
{agent_scratchpad}
---
List of work left to be done:
"""

PEER_REVIEW_TEMPLATE = """
As a data team manager, you are responsible for deciding on the next best action in service
of a stakeholder request based on the inputs of your team. Given the following stakeholder request
and team inputs, concisely describe the next best action to take and why.  Let's think step by
step. Don't suggest work which is not explicitly asked for in the request. Don't suggest visualization.
Don't suggest exporting the data from the dataframe. Now, take a deep breath and work on this step by
step.

Stakeholder request: {query}
Required Metrics: {metrics}
Required Data Dimensions: {dimensions}
Required Data Granularity: {granularity}
---
Learned context: {learned_context}
---
Data Analyst Suggested Next Steps: {data_analyst}
---
Data Engineer Suggested Next Steps: {data_engineer}
---
Data Scientist Suggested Next Steps: {data_scientist}
---
Next best action:
"""

TRANSLATE_SQL_TEMPLATE = (
    "As an expert data warehouse engineer, you are tasked with fixing an erroneous SQL statement. "
    "Given the following SQL statement and resulting error from its execution, fix the error and "
    "modify the SQL to be compliant with the GoogleSQL dialect. Always use table aliases and "
    "fully qualified column names. Now, take a deep breath and work on this step by step.\n"
    "\n"
    "SQL Statement: {sql_statement}\n"
    "BigQuery Error:\n---\n{error}\n---\n\n"
    "Output format: \n"
    "```json\n"
    '{{\n  "reasoning": $str_explanation_of_fix\n  "sql_statement": $sql_statement\n}}'
    "\n```\n\n"
    "Output: \n"
)
