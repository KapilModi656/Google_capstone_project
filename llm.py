import os
import warnings

from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
load_dotenv()
mistral_api_key = os.environ.get("MISTRAL_API_KEY")


try:
    import mistralai

    mistral_client = mistralai.Mistral(api_key=mistral_api_key)
except Exception:
   
    try:
        from mistralai.client import MistralClient as LegacyMistralClient

        warnings.warn(
            "Using legacy `mistralai.client.MistralClient`. Consider upgrading to the new `mistralai` SDK.",
            DeprecationWarning,
        )
        mistral_client = LegacyMistralClient(api_key=mistral_api_key)
    except Exception as e:
     
        raise RuntimeError(
            "Could not initialize a Mistral client.\n"
            "Install the new `mistralai` package or pin the old version: `pip install 'mistralai==0.4.2'`.\n"
            f"Original error: {e}"
        )


mistral_llm_model = LiteLlm(model="mistral/codestral-latest")


def get_mistral_llm_agent():
    return mistral_llm_model