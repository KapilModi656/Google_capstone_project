from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
from pathlib import Path
import sys
import asyncio
import numpy as np
import json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from llm import get_mistral_llm_agent
from utils import execute_code
from rl.rl_agent import main as rl_agent_main

llm = get_mistral_llm_agent()

def execute_code_tool(code: str):
    """Execute a block of Python code provided as a string and return serializable result."""
    exec_globals = {}
    return execute_code(code, exec_globals)
def rl_wrapper(budget: int = 1000, episodes: int = 20):
    """
    Runs RL agent, sanitizes output, and SAVES it to a file for easy retrieval.
    """

    raw_result = rl_agent_main(budget=budget, episodes=episodes)
    
   
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
        return obj

    clean_result = convert_to_serializable(raw_result)
    with open("rl_output.json", "w") as f:
        json.dump(clean_result, f)
        
    return clean_result
ml_trainer_agent = LlmAgent(
    name="ml_trainer_agent",
    model=llm,
    instruction=(
        "You are an expert ML Engineer. Your goal is to scale and train a model based on the provided dataset path. "
        "1. Generate Python code to load the data, scale it, and train a model (XGBoost/LightGBM/Sklearn) depend on number of rows and type of data if less rows use linear regression model if more use random forest or xgboost or lightgbm.  "
        "2. Save the model to 'file/model.pkl' and scaler to 'file/scaler.pkl'. and scaler for target column save it at 'file/y_scaler.pkl' "
        "3. Use the 'os' module to ensure the directory exists. "
        "4. IMPORTANT: The final line of your output must be the absolute paths of the saved files."
        "5. target column is named 'sales' or 'Sales'."
        "6. Drop the first column then proceed with training. with remaining columns"
    ),
    description="Execute python code to perform ML model training task",
    tools=[FunctionTool(execute_code_tool)],
)

rl_agent = LlmAgent(
    name="rl_agent",
    model=llm,
    instruction=(
        "You are an RL Specialist. Run the RL agent for budget allocation. "
        "Use the provided tools to execute the logic. You will be given context from a previous ML training step."
    ),
    description="Execute python code to perform RL budget allocation task",
    tools=[FunctionTool(rl_wrapper)],
)


async def workflow(dataset_path: str, total_budget: int = 1000, episodes: int = 20):
    """
    Orchestrates the workflow of data analysis, ML model training, and RL budget allocation.
    """
    
    
    print(f"--- Starting ML Training on {dataset_path} ---")
    ml_runner = InMemoryRunner(agent=ml_trainer_agent, plugins=[LoggingPlugin()])
    
    ml_prompt = f"Please train a model using the dataset located at: '{dataset_path}'"
    
   
    ml_result = await ml_runner.run_debug(ml_prompt)
    
    
    ml_output_text = ml_result.text if hasattr(ml_result, 'text') else str(ml_result)

    
    print(f"--- Starting RL Allocation (Budget: {total_budget}) ---")
    rl_runner = InMemoryRunner(agent=rl_agent, plugins=[LoggingPlugin()])
    
    rl_prompt = f"""
    Perform budget allocation with the following parameters:
    Total Budget: {total_budget}
    Episodes: {episodes}
    
    Context from ML Training Step:
    {ml_output_text}
    """
    
    await rl_runner.run_debug(rl_prompt)
    try:
        with open("rl_output.json", "r") as f:
            final_data = json.load(f)
    except FileNotFoundError:
        final_data = {"error": "RL Agent did not save output file."}
    return {
        "ml_summary": ml_output_text,
        "rl_result": final_data
    }

