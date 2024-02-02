"""Tool prompts for the agents."""

DATA_TABLE_SCHEMA_SEARCH = (
    "Given the following query, return curated schemas of the following database tables. "
    "Make sure to include the fully qualified table name in the curated schema. "
    "Only include fields which might be relevant to servicing the query, as well as any "
    "fields representing primary or foriegn keys. Be sure to maintain any hierarchies and "
    "structure represented in the schemas.\n\nQuery: {question}\n\nSchemas:\n{context}")

PYDANTIC_FORMAT_INSTRUCTIONS = """When using a serialized JSON object in your response,
it MUST be compliant with RFC8259 and conform to the output JSON schema below.

The output JSON schema you should use is:
```
{schema}

```
{extra_instructions}
"""
