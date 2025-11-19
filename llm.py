import os
import warnings

from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Prefer the new `mistralai.Mistral` SDK. Fall back to the old client only if needed.
try:
    import mistralai

    # New SDK exposes `Mistral` class which accepts `api_key`.
    mistral_client = mistralai.Mistral(api_key=mistral_api_key)
except Exception:
    # Try the legacy import (may be present but deprecated)
    try:
        from mistralai.client import MistralClient as LegacyMistralClient

        warnings.warn(
            "Using legacy `mistralai.client.MistralClient`. Consider upgrading to the new `mistralai` SDK.",
            DeprecationWarning,
        )
        mistral_client = LegacyMistralClient(api_key=mistral_api_key)
    except Exception as e:
        # Re-raise a clearer error explaining how to fix the environment
        raise RuntimeError(
            "Could not initialize a Mistral client.\n"
            "Install the new `mistralai` package or pin the old version: `pip install 'mistralai==0.4.2'`.\n"
            f"Original error: {e}"
        )


# The `LiteLlm` wrapper expects a `model` string. The underlying provider
# is determined by the model name and environment (e.g., `MISTRAL_API_KEY`).
# Let LiteLlm use the configured environment; pass only the model identifier.
mistral_llm_model = LiteLlm(model="mistral/codestral-latest")


def get_mistral_llm_agent():
    return mistral_llm_model