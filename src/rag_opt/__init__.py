from langchain.embeddings.base import init_embeddings as _langchain_init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain.chat_models import init_chat_model as _langchain_init_chat_model
from rag_opt.rag import init_vectorstore, init_reranker
from typing import Any, Union, Optional
from rag_opt._config import RAGConfig
from loguru import logger
import os 


def init_chat_model(
    model: str,
    *,
    model_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Union[Embeddings, Runnable[Any, list[float]]]:
    
    if model_provider == "sentence-transformers":
        model_provider = "huggingface"
        logger.warning("Using HuggingFace provider for sentence transformer models")
    
    out = None
    if model_provider == "huggingface":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
                repo_id=model,
                task="text-generation",
                max_new_tokens=1024, # TODO:: make this configurable
                do_sample=False,
                repetition_penalty=1.03,
                huggingfacehub_api_token=api_key
            )
       
        out =  ChatHuggingFace(llm=llm)
    else:
        out = _langchain_init_chat_model(model, model_provider=model_provider,api_key=api_key, **kwargs)

    logger.success(f"{model} LLM Loaded successfully")
    return out 

def init_embeddings(
    model: str,
    *,
    model_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Union[Embeddings, Runnable[Any, list[float]]]:
    if model_provider == "sentence-transformers":
        model_provider = "huggingface"
        logger.warning("Using HuggingFace provider for sentence transformer models")
    
    out = None
    if model_provider == "huggingface":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from langchain_huggingface import HuggingFaceEmbeddings

        out =  HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"}, # TODO:: make this configurable
                encode_kwargs={"normalize_embeddings": False},
            )
    else:
        out = _langchain_init_embeddings(model, provider=model_provider,api_key=api_key, **kwargs)
    logger.success(f"{model} Embeddings Loaded successfully")
    return out 

__all__ =[
    "init_embeddings",
    "init_chat_model",
    "init_vectorstore",
    "init_reranker",
    "RAGConfig"
]