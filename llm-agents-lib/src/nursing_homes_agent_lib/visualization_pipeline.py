import enum
from typing import Callable

from attr import dataclass
from dotenv import load_dotenv
from sqlalchemy import Engine, text
import polars as pl

from nursing_homes_agent_lib.plot_builder_agent import PlotBuilderAgent, PlotBuilderAgentResult, \
    plot_builder_agent_provider
from nursing_homes_agent_lib.text2sql_agent import Text2SqlAgent, Text2SqlAgentResult, text2sql_agent_builder_provider, \
    build_duckdb_engine, get_txt2sql_default_config


@dataclass
class VisualizationDataPipelineResult:
    txt2sql_agent_result: Text2SqlAgentResult
    plot_builder_agent_result: PlotBuilderAgentResult | None


@dataclass
class DfFetchResult:
    df: pl.DataFrame | None
    error: str | None


class DataPipelineStep(str, enum.Enum):
    TXT2SQL = "txt2sql"
    FETCH_DF = "fetch_df"
    PLOT_BUILDER = "plot_builder"
    END = "end"
    ERROR = "error"


class VisualizationDataPipeline:
    def __init__(
            self,
            text2sql_agent: Text2SqlAgent,
            plot_builder_agent: PlotBuilderAgent,
            db_engine: Engine,
            user_input: str,
            log_callback: Callable[[str], None] | None = None
    ):
        self.text2sql_agent = text2sql_agent
        self.plot_builder_agent = plot_builder_agent
        self.db_engine = db_engine

        self.user_input = user_input

        self.log_callback = log_callback
        self.next_step: DataPipelineStep = DataPipelineStep.TXT2SQL
        self.txt2sql_res: Text2SqlAgentResult | None = None
        self.df: pl.DataFrame | None = None

    def run_txt2sql_agent(self) -> Text2SqlAgentResult:
        if not self.next_step == DataPipelineStep.TXT2SQL:
            raise ValueError("Pipeline is not in the correct state to run the Text2SQL agent.")

        self.txt2sql_res = self.text2sql_agent.run(self.user_input, lambda msg: self._log_callback("Text2SQL", msg))

        if self.txt2sql_res.error:
            self._log_callback("Text2SQL", f"Error: {self.txt2sql_res.error}")

        self.next_step = DataPipelineStep.FETCH_DF
        return self.txt2sql_res

    def run_fetch_df(self) -> DfFetchResult:
        if not self.next_step == DataPipelineStep.FETCH_DF:
            raise ValueError("Pipeline is not in the correct state to run the fetch_df step.")

        self._log_callback("fetch DF", "Executing SQL query...")

        try:
            with self.db_engine.connect() as connection:
                result = connection.execute(text(self.txt2sql_res.generated_sql))
                column_names = list(result.keys())
                data_res = list(result.fetchall())

            self.df = pl.DataFrame(data_res, schema=column_names)
            error = None
            self.next_step = DataPipelineStep.PLOT_BUILDER
        except Exception as e:
            error = str(e)
            self.next_step = DataPipelineStep.ERROR

        return DfFetchResult(
            df=self.df,
            error=error
        )

    def run_plot_builder_agent(self):
        if not self.next_step == DataPipelineStep.PLOT_BUILDER:
            raise ValueError("Pipeline is not in the correct state to run the PlotBuilder agent.")

        plot_builder_res = self.plot_builder_agent.run(
            self.user_input,
            self.df,
            lambda msg: self._log_callback("PlotBuilder", msg)
        )

        self.next_step = DataPipelineStep.END

        return plot_builder_res

    def _log_callback(self, sub_agent, message: str):
        if self.log_callback:
            self.log_callback(f'{sub_agent}: {message}')


def main():
    load_dotenv("../../.env")
    txt2sql_config = get_txt2sql_default_config()
    db_engine = build_duckdb_engine(txt2sql_config.duckdb_file_name)
    txt2sql_agent = text2sql_agent_builder_provider().build()
    plot_builder_agent = plot_builder_agent_provider()

    user_input = "how many injuries per day per facility"

    pipeline = VisualizationDataPipeline(
        txt2sql_agent,
        plot_builder_agent,
        db_engine,
        user_input
    )

    res = pipeline.run_txt2sql_agent()

    # visualize plotly figure
    # if res.plot_builder_agent_result.plotly_fig is not None:
    #     res.plot_builder_agent_result.plotly_fig.show()

    pass


if __name__ == '__main__':
    main()
