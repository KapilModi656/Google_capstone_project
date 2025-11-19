import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rl.rl_agent import get_rl_agent
from llm import get_mistral_llm_agent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
from llm import get_mistral_llm_agent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import csv
from pathlib import Path as _Path

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.plugins.logging_plugin import LoggingPlugin

from llm import get_mistral_llm_agent
from logger import get_logger
from utils import execute_code

llm = get_mistral_llm_agent()
logging = get_logger(__name__)


def execute_code_tool(code, exec_globals=None):
    """ADK-friendly wrapper around `utils.execute_code`.

    Avoid type annotations and mutable default values to prevent ADK
    from trying to JSON-serialize Python `type` objects.
    """
    if exec_globals is None:
        exec_globals = {}
    return execute_code(code, exec_globals)

from llm import get_mistral_llm_agent
from logger import get_logger
from utils import execute_code

llm = get_mistral_llm_agent()
logging = get_logger(__name__)



def execute_code_tool(code, exec_globals=None):
    """ADK-friendly wrapper around `utils.execute_code`.

    Avoid type annotations and mutable default values to prevent ADK
    from trying to JSON-serialize Python `type` objects.
    """
    if exec_globals is None:
        exec_globals = {}
    return execute_code(code, exec_globals)


# LLM agents: they will run small code snippets via `execute_code_tool`.
# We keep the tools minimal and avoid complex annotated signatures.
data_analyst_agent = LlmAgent(
    name="data_analyst_agent",
    model=llm,
    instruction=(
        "You are a ML data analyst. You have been provided with a dataset and you need to analyze it "
        "and provide insights. Write a Python code block only and print results after performing any operation."
    ),
    description="Execute python code to perform data analytics task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='data_analytics_output'
)


data_transformation_agent = LlmAgent(
    name="data_transformation_agent",
    model=llm,
    instruction=(
        "You are a data transformer. You have been provided with a dataset and you need to transform it. "
        "Write a Python code block only and print top 5 rows after each transformation. Save transformed dataset and print its path."
    ),
    description="Execute python code to perform data transformation task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='data_transformation_output'
)


ml_trainer_agent = LlmAgent(
    name="ml_trainer_agent",
    model=llm,
    instruction=(
        "You are a ML model trainer. Train a model on the transformed dataset, scale features with StandardScaler, "
        "save the model and scaler under `file_build/` using pickle, and print saved file paths."
    ),
    description="Execute python code to perform ML model training task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='ml_trainer_output'
)


prediction_pipeline = LlmAgent(
    name="prediction_pipeline_agent",
    model=llm,
    instruction=(
        "Create a prediction pipeline that loads the trained model and scaler (from `file_build/`) and defines "
        "a function `prediction_function(observation)` which returns a prediction. Then call it on TEST_OBSERVATION=[[5.1,3.5,1.4]] "
        "and print the result with the prefix 'PREDICTION_OUTPUT:'."
    ),
    description="Execute python code to create prediction pipeline",
    tools=[FunctionTool(execute_code_tool)],
    output_key='prediction_pipeline_output'
)


rl_agent = LlmAgent(
    name="rl_agent",
    model=llm,
    instruction=(
        "You are a reinforcement learning agent. Use the prediction pipeline output to suggest improvements. "
        "(Note: RL will be executed locally after ADK artifacts are produced.)"
    ),
    description="Execute python code to perform reinforcement learning task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='rl_agent_output'
)


def run_data_analyst(dataset_path: str = None, total_budget: int = 1000, episodes: int = 20, action_space: int = 10):
    """Run the ADK pipeline for `dataset_path` and then run local RL to allocate `total_budget`.

    Returns a dict with keys: `adk_result`, `rl_result`.
    """
    agent = SequentialAgent(
        name="data_pipeline_sequential",
        sub_agents=[
            data_analyst_agent,
            data_transformation_agent,
            ml_trainer_agent,
            prediction_pipeline,
            rl_agent
        ],
        description="Data Analyst to Prediction Pipeline Sequential Agent",
    )
    root = LlmAgent(
        name="root_agent",
        model=llm,
        instruction=(
            "You are the root agent coordinating the data analysis and ML pipeline. Build the pipeline and save artifacts."
        ),
        description="Root agent for coordinating the sequential agent",
        tools=[AgentTool(agent)],
    )
    runner = InMemoryRunner(agent=root, plugins=[LoggingPlugin()])

    # Read the dataset header and include column names in the prompt so agents
    # generate code using the actual column names (avoids KeyError from wrong names).
    if dataset_path is None:
        dataset_path_obj = _Path(PROJECT_ROOT) / 'data' / 'sample.csv'
    else:
        dataset_path_obj = _Path(dataset_path)

    columns = []
    try:
        with open(dataset_path_obj, newline='') as fh:
            reader = csv.reader(fh)
            columns = next(reader)
    except Exception:
        columns = []

    prompt = (
        f"Path to dataset is: '{dataset_path_obj}'. Columns: {columns}. "
        "Analyze the data, transform it, train a ML model and create a prediction pipeline. "
        "Find out the output for the provided test observation and save model artifacts."
    )

    adk_result = asyncio.run(runner.run_debug(prompt))
    print("Final ADK Result:", adk_result)

    # After ADK run completes, run the local RL runner to derive budget allocation
    try:
        # Import locally to avoid ADK automatic function-calling and serialization issues
        from rl.run_rl_with_pipeline import main as run_rl_main

        print(f"Running local RL runner to allocate total budget = {total_budget}")
        rl_result = run_rl_main(total_budget=total_budget, episodes=episodes, action_space=action_space)
        allocation = rl_result.get("allocation")
        print("Final Budget Allocation (by service):", allocation)
        print("Sum:", sum(allocation), "Total budget:", rl_result.get("total_budget"))
        # Print expected sales vs historical
        pred = rl_result.get("predicted_sales")
        hist = rl_result.get("historical_mean_sales")
        delta = rl_result.get("delta")
        pct = rl_result.get("pct_change")
        print("Predicted sales for this allocation:", pred)
        print("Historical mean sales:", hist)
        print("Delta:", delta)
        print("Percent change:", pct)
    except Exception as e:
        print("Failed to run local RL runner:", e)
        rl_result = None

    return {"adk_result": adk_result, "rl_result": rl_result}


if __name__ == "__main__":
    # Default call when script run directly
    run_data_analyst()

    


