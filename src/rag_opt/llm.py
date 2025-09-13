from langchain.chat_models import init_chat_model
from langchain_core.language_models import (
    BaseChatModel,
)


class RAGLLM(BaseChatModel):
   """
   Wrapper for Langchain chat models
   """
   