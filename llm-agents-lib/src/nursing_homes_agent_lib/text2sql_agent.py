import datetime as dt
import enum
import pathlib
import subprocess
import textwrap
from dataclasses import dataclass
from os import environ
from typing import *

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import ListSQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
from pydantic import *
from sqlalchemy import create_engine, Engine


class AggregationPeriod(str, enum.Enum):
    """Enum for the aggregation period."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class State(MessagesState):
    user_question: Annotated[str, "The user question"]
    aggregation_period: Annotated[AggregationPeriod | None, "The aggregation period"]
    tables: Annotated[List[str] | None, "The list of tables in the database"]
    table_schemas: Annotated[str | None, "The table schemas"]
    generated_sql: Annotated[str | None, "The generated SQL query"]
    number_of_attempts: Annotated[int | None, "The number of attempts to generate the SQL query"]
    error: Annotated[str | None, "The error message if the SQL generation fails"]
    callback: Annotated[Callable[[str], None] | None, "The callback function to call for trace logging"]


@dataclass
class Text2SqlAgentResult:
    generated_sql: str | None
    error: str | None


class GeneratedSql(BaseModel):
    """Generate the requested SQL for the user."""

    sql_query: str = Field(..., description="The generated SQL")
    error: str | None = Field(
        None,
        description="Error message if cannot generate a sql query"
    )
    can_generate_sql_from_db: bool = Field(
        True,
        description="True if the input question is a valid SQL request of the provided data, False otherwise."
    )


class IsValidSqlRequest(BaseModel):
    """Check if the input question is a valid SQL request of the provided data."""

    is_valid_sql: bool = Field(
        description="True if the input question is a valid SQL request that can be answered from the provided schemas, False otherwise.",
    )
    error: str | None = Field(
        None,
        description="Error message if the input question is not a valid SQL request of the provided data."
    )


class IsRequestSqlPromptAnswer(BaseModel):
    """Check if the input question is a valid SQL request of the provided data."""

    is_request_sql: bool = Field(
        ...,
        description="True if the input question is a valid SQL request of the provided data, False otherwise.",
    )
    error: str = Field(
        ...,
        description="Error message if the input question is not a valid SQL request of the provided data."
    )


LIST_TABLE_NODE = "list_tables"
GET_SCHEMA_NODE = "get_schema"
ADD_CONTEXT_NODE = "add_context"
IS_VALID_REQUEST_NODE = "is_valid_request"
QUERY_GEN_NODE = "query_gen"
VALIDATE_GEN_SQL_NODE = "validate_gen_sql"


class SqlDialect(str, enum.Enum):
    """Enum for the SQL dialect."""

    DUCKDB = "DuckDb"


@dataclass
class Text2SqlAgentConfig:
    duckdb_file_name: str = environ.get("DUCKDB_FILE_NAME")
    db_name: str = "nursing_home_db"
    db_schema: str = "nursing_home_data"
    # md_token: str = environ.get("MOTHERDUCK_TOKEN")


def build_duckdb_engine(duckdb_file_name):
    duckdb_file_path = pathlib.Path(__file__).parent / duckdb_file_name
    return create_engine(f'duckdb:///{duckdb_file_path}?access_mode=READ_ONLY')


class Text2SqlAgent:

    def __init__(self, graph: CompiledGraph):
        self.graph = graph

    def run(
            self,
            question: str,
            log_callback=None
    ) -> Text2SqlAgentResult:
        """
        Ask a question on the text2sql graph.
        :param question: user question of the data
        :param graph: text2sql graph
        :param log_callback: callback function to call for trace logging
        :return: generated SQL query
        """
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        res = self.graph.invoke({
            "user_question": question,
            "aggregation_period": None,
            "callback": log_callback,
        },
            config,
            # stream_mode="values",
        )

        return Text2SqlAgentResult(
            res.get("generated_sql"),
            res.get("error")
        )


class Text2SqlAgentBuilder:

    def __init__(self, config: Text2SqlAgentConfig, db_engine: Engine):
        self.dialect = SqlDialect.DUCKDB
        self.sql_specific_instructions_injector = _get_duckdb_specific_sql_instructions_fn
        self.db_schema = config.db_schema

        self.db_name = config.db_name
        self.engine = db_engine
        self.db = SQLDatabase(
            engine=self.engine,
            schema=self.db_schema,
            ignore_tables=["_dlt_loads", "_dlt_pipeline_state", "_dlt_version"]
        )

        # self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0)
        self.llm = init_chat_model("gemini-2.5-flash-preview-04-17", model_provider="google_genai", temperature=0)

        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        self.list_tables_tool: ListSQLDatabaseTool = next(
            tool for tool in self.tools if isinstance(tool, ListSQLDatabaseTool))

    def build(self) -> Text2SqlAgent:
        workflow = StateGraph(State)

        workflow.set_entry_point(LIST_TABLE_NODE)
        workflow.add_node(LIST_TABLE_NODE, self._list_tables_node)
        workflow.add_node(GET_SCHEMA_NODE, self._get_schema_node)
        workflow.add_node(ADD_CONTEXT_NODE, self._add_context)
        workflow.add_node(QUERY_GEN_NODE, self._query_gen_node)
        workflow.add_node(VALIDATE_GEN_SQL_NODE, self._validate_generated_sql_node)

        graph = workflow.compile()

        return Text2SqlAgent(graph)

    def _list_tables_node(self, state: State) -> Command[Literal[END, GET_SCHEMA_NODE]]:
        """
        List all the tables in the database.
        """
        question = state['user_question']

        tool_call_id = "call_list_tables_tool"

        if not question:
            res = "Error: The input question is empty. Please provide a valid question."
            return Command(
                update={
                    "messages":
                        ToolMessage(content=res, tool_call_id=tool_call_id)
                },
                goto=END
            )

        self._log(state, "Listing tables in the database...")

        table_names = [t.strip() for t in self.db.get_usable_table_names()]

        if len(table_names) == 0:
            res = "Error: No tables found in the database."
            return Command(
                update={
                    "messages":
                        ToolMessage(content=res, tool_call_id=tool_call_id)
                },
                goto=END
            )

        return Command(
            update={"tables": table_names},
            goto=GET_SCHEMA_NODE
        )

    def _get_schema_node(self, state: State, ) -> Command[Literal[END, ADD_CONTEXT_NODE]]:
        table_names = state["tables"]

        if not table_names:
            res = "Error: No tables found in the database."
            msg = SystemMessage(
                content=res,
            )
            return Command(
                update={"messages": [msg]},
                goto=END
            )

        self._log(state, "Getting table schemas...")

        schemas_str = self.db.get_table_info_no_throw(table_names)

        if "Error:" in schemas_str:
            return Command(
                update={"messages":
                    SystemMessage(
                        content=f"Error: {schemas_str}",
                    )
                },
                goto=END
            )

        if not schemas_str:
            res = "Error: No table schemas found in the database."
            return Command(
                update={
                    "messages":
                        SystemMessage(
                            content=res,
                            # tool_call_id=tool_call_id
                        )
                },
                goto=END
            )

        return Command(goto=ADD_CONTEXT_NODE,
                       update={"table_schemas": schemas_str})

    def _add_context(self, _: State) -> Command[Literal[QUERY_GEN_NODE]]:
        today_date = dt.datetime.today()
        msg = SystemMessage(
            content=f"Today is '{today_date.isoformat()}'",
            # tool_call_id="add_context_tool"
        )
        return Command(
            update={"messages": [msg]},
            goto=QUERY_GEN_NODE
        )

    def check_user_sql_request_node(self, state: State) -> Command[Literal[END, QUERY_GEN_NODE]]:

        full_template = """"is the user input a question? if so, can you generate a SQL query to answer it?
            
            It is not an error if a schema column type is VARCHAR but the type needs to be CAST to another type.
            
            below are the table DDL (schemas) of the database:

            {table_schemas}"""

        chat_template = ChatPromptTemplate([
            ('system', full_template),
            ('placeholder', '{messages}')
        ])

        self._log(state, "Checking if the user input is a valid SQL request...")

        step_1_res = chat_template.invoke(state)

        # formatted_prompt_step = ChatPromptTemplate.from_messages(
        #     [("system", _build_is_request_for_sql_prompt()), ("user", "{messages}"), ])

        # step_1_res = formatted_prompt_step.invoke({
        #     "messages": state["messages"]
        # })

        self._log(state, "Checking if the user input is a valid SQL request of the provided data...")

        step_2_res: IsValidSqlRequest = self.llm.with_structured_output(IsValidSqlRequest).invoke(step_1_res)

        if not step_2_res.is_valid_sql:
            return Command(
                update={
                    "messages":
                        AIMessage(
                            content=f"Error: {step_2_res.error}\n please fix your mistakes."
                        )
                },
                goto=END
            )

        return Command(goto=QUERY_GEN_NODE)

    def _query_gen_node(self, state: State) -> Command[Literal[END, QUERY_GEN_NODE, VALIDATE_GEN_SQL_NODE]]:

        attempt_number = state.get("number_of_attempts", 0) + 1

        messages_ = []

        if attempt_number > 1:
            messages_.append(
                SystemMessage(
                    "this is a subsequent attempt to generate a query, please fix the previously invalid generated query"
                )
            )

            prev_query = state.get("generated_sql")

            messages_.append(
                SystemMessage(
                    f"previous generated query: {prev_query}"
                )
            )

        messages_.append(MessagesPlaceholder("messages"))

        messages_.append(
            HumanMessage(_build_query_gen_system_prompt(
                self.dialect,
                self.sql_specific_instructions_injector,
                state["user_question"],
                state["tables"],
                state["table_schemas"],
                state["aggregation_period"]
            ))
        )

        out1 = ChatPromptTemplate(messages_).invoke({"messages": state["messages"]})

        # query_gen = (
        #         ChatPromptTemplate(messages_)
        #         | self.llm.with_structured_output(GeneratedSql)
        # )

        query_gen = self.llm.with_structured_output(GeneratedSql)

        if attempt_number > 1:
            log_msg = f"Generating SQL query for attempt {attempt_number}..."
        else:
            log_msg = f"Generating SQL query..."
        self._log(state, log_msg)

        res_msg: GeneratedSql = query_gen.invoke(out1)

        update_dict = {
            "number_of_attempts": attempt_number,
        }

        if not res_msg.can_generate_sql_from_db:
            return Command(
                update={
                    **update_dict,
                    'error': f"{res_msg.error}"
                },
                goto=END
            )

        if res_msg.error is not None:
            if attempt_number >= 3:
                return Command(
                    update={
                        **update_dict,
                        "error": f"Maximum number of attempts reached generating SQL. Error: {res_msg.error}.",
                    },
                    goto=END
                )
            return Command(
                update={
                    **update_dict,
                    "messages": AIMessage(
                        content=f"Error: {res_msg.error}"
                    )
                },
                goto=QUERY_GEN_NODE
            )

        return Command(
            update={
                **update_dict,
                "generated_sql": res_msg.sql_query
            },
            goto=VALIDATE_GEN_SQL_NODE
        )

    def _validate_generated_sql_node(self, state: State) -> Command[Literal[END, QUERY_GEN_NODE]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        generated_sql = state["generated_sql"]
        if not generated_sql.strip():
            return Command(
                update={
                    "messages":
                        AIMessage(
                            content="Error: The generated SQL query is empty. Please provide a valid SQL query to check."
                        )
                },
                goto=END
            )

        self._log(state, "Validating generated SQL query against db...")

        try:
            test_sql_res = self.db.run(f'EXPLAIN {generated_sql}')
        except Exception as e:
            return Command(
                update={
                    "messages":
                        AIMessage(
                            content=f"Error: {repr(e)}"
                        )
                },
                goto=QUERY_GEN_NODE
            )

        return Command(goto=END)

    @staticmethod
    def _log(state: State, message: str):
        if state["callback"] is not None:
            state["callback"](message)


def text2sql_agent_builder_provider(db_engine=None) -> Text2SqlAgentBuilder:
    config = get_txt2sql_default_config()
    if db_engine is None:
        db_engine = build_duckdb_engine(config.duckdb_file_name)
    return Text2SqlAgentBuilder(config, db_engine)

def _get_duckdb_specific_sql_instructions_fn() -> list[str]:
    return [
        f"if merging two tables on a timestamp column, and the join is an inequality, you can use ASOF JOIN. example: nurse response time after an incident, response time >= incident time",
        "if required to do a function(INTERVAL), use EXTRACT(EPOCH FROM INVERVAL) instead"
    ]


def _build_query_gen_system_prompt(
        dialect: SqlDialect,
        instruction_getter_fn: Callable[[], list[str]],
        question: str,
        tables: List[str],
        table_schemas: str,
        aggregation_period: AggregationPeriod | None = None
) -> str:
    instructions = [
        "if the user input is not a question or request of the data, return an error message.",
        "if multiple requests of the data are present, return an error message.",
        f"output a syntactically correct {dialect.value} query to run.",
        "prefer CTEs over subqueries.",
        *instruction_getter_fn(),
        f"Make sure to only reference the following DB tables: {", ".join(tables)}.",
    ]

    if aggregation_period is not None:
        instructions.append(
            f"Aggregate the results by {aggregation_period}, and if not possible, return an error message."
        )

    instructions.extend((
        # "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.",
        "Unless specified, order the results by a relevant column to return the most interesting examples in the database.",
        "only query for relevant columns given the user request.",
        "NEVER make stuff up if you don't have enough information to answer the query.",
        "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.",
    ))

    p1 = f"""instructions:"""

    for instruction in instructions:
        p1 += "\n\n- " + instruction

    p1 += f"""\n\n- use the following db schema tables as context:

    {table_schemas}


    user input: 
    {question}"""

    return textwrap.dedent(p1)


def _build_query_check_system_prompt() -> str:
    return f"""You are a SQL expert with a strong attention to detail.
Double check the sql query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""


def get_txt2sql_default_config() -> Text2SqlAgentConfig:
    return Text2SqlAgentConfig()


def print_graph():
    config = get_txt2sql_default_config()
    db_engine = build_duckdb_engine(config.duckdb_file_name)
    agent_builder = Text2SqlAgentBuilder(config, db_engine)
    agent = agent_builder.build()
    agent.graph.get_graph().draw_mermaid_png(output_file_path="../../graph.png")
    # open file using default image viewer
    subprocess.run(["start", "graph.png"], shell=True)


def test_agent():
    config = get_txt2sql_default_config()
    db_engine = build_duckdb_engine(config.duckdb_file_name)
    agent_builder = Text2SqlAgentBuilder(config, db_engine)
    # question = "What is the average age of patients?"
    while question := input("ask a question of the data...\n"):
        agent = agent_builder.build()
        agent.run(question)
    print("Exited...")
    pass


if __name__ == '__main__':
    load_dotenv()

    # print_graph()
    test_agent()
