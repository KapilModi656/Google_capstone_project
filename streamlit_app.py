import streamlit as st
import tempfile
from pathlib import Path
import os

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
            # Save uploaded file to workspace data dir
            repo_root = Path(__file__).resolve().parents[0]
            data_dir = repo_root / "data"
            data_dir.mkdir(exist_ok=True)
            save_path = data_dir / "uploaded_dataset.csv"
            with open(save_path, "wb") as fh:
                fh.write(uploaded.getbuffer())

            st.info(f"Saved uploaded dataset to {save_path}")

            # Import the pipeline runner
            try:
                # Ensure project root on path
                import sys
                sys.path.insert(0, str(repo_root))
                from AI_ML_BUILDER.src.data_analyst_fixed import run_data_analyst

                result = run_data_analyst(str(save_path), total_budget=int(budget), episodes=int(episodes), action_space=10)

                st.success("Pipeline completed")
                rl_result = result.get("rl_result")
                if rl_result is None:
                    st.warning("RL runner failed to produce a result; check logs.")
                else:
                    st.subheader("Budget Allocation")
                    allocation = rl_result.get("allocation")
                    st.write(allocation)
                    st.write(f"Sum: {sum(allocation)} / Total budget: {rl_result.get('total_budget')}")

                    st.subheader("Sales comparison")
                    st.write("Predicted sales for allocation:", rl_result.get("predicted_sales"))
                    st.write("Historical mean sales:", rl_result.get("historical_mean_sales"))
                    st.write("Delta:", rl_result.get("delta"))
                    st.write("Pct change:", rl_result.get("pct_change"))

            except Exception as e:
                st.exception(e)
