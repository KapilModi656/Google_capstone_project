import streamlit as st
import tempfile
from pathlib import Path
import os
import asyncio
import nest_asyncio
from llm import get_mistral_llm_agent
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
llm = ChatMistralAI(model="codestral-latest", api_key=mistral_api_key)

nest_asyncio.apply()
st.set_page_config(page_title="ML + RL Pipeline Runner", layout="wide")
st.title("ML + RL Pipeline — Upload dataset and run")

st.markdown("Upload a CSV dataset (first row must be headers). The app will run the ADK pipeline and then a local RL runner that returns a budget allocation summing to the total budget.")

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
budget = st.number_input("Total budget (integer)", min_value=0, value=1000, step=100)
episodes = st.number_input("RL episodes (short runs recommended)", min_value=1, value=20, step=1)

if st.button("Run pipeline"):
    if uploaded is None:
        st.error("Please upload a CSV file before running.")
    else:
        with st.spinner("Saving upload and running pipeline. This may take a while (LLM calls)..."):
           
            repo_root = Path(__file__).resolve().parents[0]
            data_dir = repo_root / "data"
            data_dir.mkdir(exist_ok=True)
            save_path = data_dir / "dataset.csv"
            with open(save_path, "wb") as fh:
                fh.write(uploaded.getbuffer())

            st.info(f"Saved uploaded dataset to {save_path}")

            try:
               
                import sys
                sys.path.insert(0, str(repo_root))
                from AI_ML_BUILDER.src.Data_Analyst import workflow

                result = asyncio.run(workflow(str(save_path), total_budget=int(budget), episodes=int(episodes)))

                st.success("Pipeline completed")
                rl_result = result.get("rl_result")
                if rl_result is None:
                    st.warning("RL runner failed to produce a result; check logs.")
                else:
                    st.subheader("RL Budget Allocation Result")
                  
                    prompt=f"""
You are an expert data analyst. and master of Marketing budget allocation. RL agent has produced the following budget allocation result based on a pre-trained ML model.
Provide a concise analysis of the following RL budget allocation result:
{rl_result}
"""
                    chain = llm | StrOutputParser()
                    analysis = chain.invoke(prompt)
                    st.markdown(analysis)

            except Exception as e:
                st.exception(e)
