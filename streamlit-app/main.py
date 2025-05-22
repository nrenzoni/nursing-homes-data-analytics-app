import time

import streamlit as st
from dotenv import load_dotenv
from nursing_homes_agent_lib.text2sql_agent import (
    build_duckdb_engine,
    text2sql_agent_builder_provider
)
from nursing_homes_agent_lib.plot_builder_agent import (
    plot_builder_agent_provider,
    build_default_plot_builder_agent_config, VizType
)
from nursing_homes_agent_lib.visualization_pipeline import VisualizationDataPipeline

load_dotenv()

st.title("Nursing Home Analytics App")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using Streamlit's native chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message")


def get_visualization_data_pipeline(user_input: str, log_callback=None):
    config = build_default_plot_builder_agent_config()
    db_engine = build_duckdb_engine(config.duckdb_file_name)

    text_sql_agent_builder = text2sql_agent_builder_provider(db_engine)
    text2sql_agent = text_sql_agent_builder.build()

    viz_code_builder_agent = plot_builder_agent_provider()

    visualization_data_pipeline = VisualizationDataPipeline(
        text2sql_agent,
        viz_code_builder_agent,
        db_engine,
        user_input,
        log_callback
    )

    return visualization_data_pipeline


# using a func for early stopping using return
def run(user_input: str):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        def __write_and_store_session_state(content):
            if content is None:
                return
            st.session_state.messages.append({"role": "bot", "content": content})
            st.write(content)

        def __write_and_store_session_state_simulated_typewriter(msg: str):
            if msg is None:
                return

            st.session_state.messages.append({"role": "bot", "content": msg})

            def stream_py_code():
                for word in msg.split(' '):
                    yield word + " "
                    # simulate typewriter effect
                    time.sleep(0.01)

            st.write_stream(stream_py_code)

        with st.status('Thinking...', expanded=True) as status:
            def progress_callback(msg):
                status.update(label=msg)
                st.write(msg)

            pipeline = get_visualization_data_pipeline(user_input, progress_callback)

            txt2sql_res = pipeline.run_txt2sql_agent()

            if error := txt2sql_res.error:
                with st.chat_message("bot"):
                    __write_and_store_session_state(f"Error: {error}")
                return

            with st.chat_message("bot"):
                __write_and_store_session_state_simulated_typewriter(
                    f"SQL: {txt2sql_res.generated_sql}"
                )

            df_res = pipeline.run_fetch_df()

            if error := df_res.error:
                with st.chat_message("bot"):
                    __write_and_store_session_state(f"Error: {error}")
                return

            __write_and_store_session_state(df_res.df)

            plot_builder_res = pipeline.run_plot_builder_agent()

            if plot_builder_res.py_plotly_code:
                with st.chat_message("bot"):
                    __write_and_store_session_state_simulated_typewriter(
                        f'py code:\n{plot_builder_res.py_plotly_code}'
                    )

            if error := plot_builder_res.error:
                with st.chat_message("bot"):
                    __write_and_store_session_state(f"Error: {error}")
                return

            if security_warning := plot_builder_res.security_warning:
                with st.chat_message("bot"):
                    __write_and_store_session_state(f"Security Warning: {security_warning}")
                return

            status.update(label='Done!')

        if plot_builder_res.viz_type == VizType.TABLE:
            # no plot since already displayed dataframe
            return

        if plotly_fig := plot_builder_res.plotly_fig:
            with st.chat_message("bot"):
                __write_and_store_session_state(plotly_fig)
            return

        # shouldn't ever get here
        with st.chat_message("bot"):
            __write_and_store_session_state("No valid response received.")


run(user_input)
