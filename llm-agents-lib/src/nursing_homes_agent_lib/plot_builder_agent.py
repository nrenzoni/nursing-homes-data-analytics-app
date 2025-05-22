import enum
from dataclasses import dataclass
from os import environ
from typing import Callable

import plotly
import plotly.graph_objects
import polars as pl
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


@dataclass
class VizCodeBuilderAgentConfig:
    duckdb_file_name: str


class VizType(str, enum.Enum):
    TABLE = "table"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"


class VizStructuredOutputModel(BaseModel):
    python_code: str = Field(...)
    viz_type: VizType = Field(description="type of the plot. if table type, do not generate plot code.")
    error: str = Field(..., description="error message if cannot generate the python code")


class SecurityCheckLlmResult(BaseModel):
    is_security_risk: bool
    security_warning: str = Field(
        ...,
        description="security warning message if code poses a security risk, otherwise blank")


@dataclass
class PlotBuilderAgentResult:
    viz_type: VizType
    py_plotly_code: str | None
    plotly_fig: plotly.graph_objects.Figure | None
    error: str | None
    security_warning: str | None


class PlotBuilderAgent:

    def __init__(
            self,
            config: VizCodeBuilderAgentConfig
    ):
        self._llm = init_chat_model("gemini-2.5-flash-preview-04-17", model_provider="google_genai", temperature=0)

    def run(
            self,
            user_query: str,
            df: pl.DataFrame,
            callback: Callable[[str], None] = None
    ) -> PlotBuilderAgentResult:
        """
        Run the agent with the given user query.
        """

        def __log(message: str):
            if callback:
                callback(message)

        __log("Generating plotly code...")

        llm_res: VizStructuredOutputModel = self._llm.with_structured_output(VizStructuredOutputModel).invoke(
            f"""
            - Generate concise python code which builds a plotly graph, if relevant.
            - if the user query is about a table, or the ideal visualization method is a table, set viz_type = 'table'.
            - if viz_type is 'table', do not generate any plotly code.
            - the code should reference df (a polars dataframe), which has the following format and sample data:
            
            {df.head(10)} 
            
            - the code should store the plotly figure in a variable called plotly_fig.
            - if not possible, don't generate code, and return an error message.
            - leave out comments from the code.
            - if aggregating over time, prefer line plot (time should be on x-axis).
            - if line plot is requested:
                - if most groups have at most 1 data point, then draw a scatter chart.
                - otherwise if some groups have only 1 data point, exclude those groups from the line plot.
                - otherwise, draw a line plot.
            - make sure to include all necessary imports, including import polars as pl.
            
            for reference the user prompt which generated the dataframe is:
            
            {user_query}"""
        )

        if error := llm_res.error:
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                llm_res.python_code,
                None,
                error,
                None
            )

        # no python code for table type
        if llm_res.viz_type == VizType.TABLE and not llm_res.python_code:
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                None,
                None,
                None,
                None
            )

        __log("Checking code for security risks...")

        security_check_res: SecurityCheckLlmResult = self._llm.with_structured_output(SecurityCheckLlmResult).invoke(
            f"""
            is the following LLM-generated python code a security risk?
            security risk examples:
            - connects to internet
            - retrieves files

            generated code:
            
            {llm_res.python_code}
""")

        if security_warning := security_check_res.security_warning:
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                None,
                None,
                None,
                f"security error: {security_warning}"
            )

        if security_check_res.is_security_risk:
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                llm_res.python_code,
                None,
                None,
                "the generated code poses a security risk"
            )

        # passed the above simple security risk checks...

        __log("Executing generated plotly code...")

        local_vars = {"df": df}

        try:
            exec(llm_res.python_code, {}, local_vars)
        except Exception as e:
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                llm_res.python_code,
                None,
                str(e),
                None
            )

        plotly_fig = local_vars.get("plotly_fig")

        if not isinstance(plotly_fig, plotly.graph_objects.Figure):
            return PlotBuilderAgentResult(
                llm_res.viz_type,
                llm_res.python_code,
                None,
                f"expected plotly.Figure type, instead got {type(plotly_fig)}",
                None
            )

        return PlotBuilderAgentResult(
            llm_res.viz_type,
            llm_res.python_code,
            plotly_fig,
            None,
            None
        )


def build_default_plot_builder_agent_config():
    return VizCodeBuilderAgentConfig(
        duckdb_file_name=environ["DUCKDB_FILE_NAME"]
    )


def plot_builder_agent_provider() -> PlotBuilderAgent:
    config = build_default_plot_builder_agent_config()
    return PlotBuilderAgent(config)
