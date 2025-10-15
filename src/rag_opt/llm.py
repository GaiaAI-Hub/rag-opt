from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class RAGLLM(BaseChatModel):
   """
   Wrapper for Langchain chat models
   """

class RAGEmbedding(Embeddings):
   """
   Wrapper for Langchain embeddings
   """