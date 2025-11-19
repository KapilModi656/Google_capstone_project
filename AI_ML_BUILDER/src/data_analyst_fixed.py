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


def execute_code_tool(code: str):
    """Execute a block of Python code provided as a string and return serializable result."""
    exec_globals = {}
    return execute_code(code, exec_globals)


# Minimal ADK agents using the safe wrapper above
data_analyst_agent = LlmAgent(
    name="data_analyst_agent",
    model=llm,
    instruction="Analyze the dataset and print results (only Python code block).",
    description="Execute python code to perform data analytics task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='data_analytics_output'
)

# Reuse same simple agent for the rest of pipeline steps in this fixed module
data_transformation_agent = LlmAgent(
    name="data_transformation_agent",
    model=llm,
    instruction="Transform the dataset and save transformed version to disk.",
    description="Execute python code to perform data transformation task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='data_transformation_output'
)

ml_trainer_agent = LlmAgent(
    name="ml_trainer_agent",
    model=llm,
    instruction="Train a model and save artifacts under file_build/ using pickle.",
    description="Execute python code to perform ML model training task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='ml_trainer_output'
)

prediction_pipeline = LlmAgent(
    name="prediction_pipeline_agent",
    model=llm,
    instruction="Create a prediction function that loads saved model and predicts for TEST_OBSERVATION=[[5.1,3.5,1.4]].",
    description="Execute python code to create prediction pipeline",
    tools=[FunctionTool(execute_code_tool)],
    output_key='prediction_pipeline_output'
)

rl_agent = LlmAgent(
    name="rl_agent",
    model=llm,
    instruction="Suggest RL improvements; RL will be executed locally after ADK artifacts are produced.",
    description="Execute python code to perform reinforcement learning task",
    tools=[FunctionTool(execute_code_tool)],
    output_key='rl_agent_output'
)


def run_data_analyst(dataset_path: str = None, total_budget: int = 1000, episodes: int = 20, action_space: int = 10):
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
        instruction="Coordinate pipeline and save artifacts.",
        description="Root agent for coordinating the sequential agent",
        tools=[AgentTool(agent)],
    )
    runner = InMemoryRunner(agent=root, plugins=[LoggingPlugin()])

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

    prompt = f"Path: {dataset_path_obj}. Columns: {columns}. Run the pipeline and save artifacts."

    adk_result = asyncio.run(runner.run_debug(prompt))

    # run RL locally
    try:
        from rl.run_rl_with_pipeline import main as run_rl_main
        rl_result = run_rl_main(total_budget=total_budget, episodes=episodes, action_space=action_space)
    except Exception as e:
        logging.exception("RL runner failed")
        rl_result = None

    return {"adk_result": adk_result, "rl_result": rl_result}
