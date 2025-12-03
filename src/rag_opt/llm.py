from typing import Optional, Any, Union
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models.base import BaseChatModel
from langchain.schema.embeddings import Embeddings
from loguru import logger
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
import os

 
class RAGLLM(BaseChatModel):
   """
   Wrapper for Langchain chat models
   """

class RAGEmbedding(Embeddings):
   """
   Wrapper for Langchain embeddings
   """

class VercelGatewayClient:
    """Unified client for Vercel AI Gateway"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('AI_GATEWAY_API_KEY')
        if not self.api_key:
            raise ValueError("AI_GATEWAY_API_KEY not found in environment or parameters")
        
        self.base_url = 'https://ai-gateway.vercel.sh/v1'
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def list_models(self) -> dict[str, Any]:
        """List available models through gateway"""
        try:
            models = self._client.models.list()
            return models.model_dump()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {"data": []}
    
    def get_available_llms(self) -> list[str]:
        """Get list of available LLM models"""
        models = self.list_models()
        return [m['id'] for m in models.get('data', []) if 'embedding' not in m['id'].lower()]
    
    def get_available_embeddings(self) -> list[str]:
        """Get list of available embedding models"""
        models = self.list_models()
        return [m['id'] for m in models.get('data', []) if 'embedding' in m['id'].lower()]


class VercelGatewayLLM:
    """Factory for creating LangChain LLMs via Vercel Gateway"""
    
    @staticmethod
    def create(
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create LangChain ChatOpenAI instance configured for Vercel Gateway
        
        Args:
            model: Model identifier (e.g.,'openai/gpt-4o-mini')
            api_key: Vercel AI Gateway API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LangChain BaseChatModel instance
        """
        api_key = api_key or os.getenv('AI_GATEWAY_API_KEY')
        if not api_key:
            raise ValueError("AI_GATEWAY_API_KEY required")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url='https://ai-gateway.vercel.sh/v1',
            temperature=temperature,
            max_tokens=max_tokens,
            # **kwargs # TODO:: need to check
        )


class VercelGatewayEmbeddings:
    """Factory for creating LangChain Embeddings via Vercel Gateway"""
    
    @staticmethod
    def create(
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Embeddings:
        """
        Create LangChain OpenAIEmbeddings instance configured for Vercel Gateway
        
        Args:
            model: Embedding model identifier (e.g., 'openai/text-embedding-3-small')
            api_key: Vercel AI Gateway API key
            
        Returns:
            LangChain Embeddings instance
        """
        api_key = api_key or os.getenv('AI_GATEWAY_API_KEY')
        if not api_key:
            raise ValueError("AI_GATEWAY_API_KEY required")
        
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url='https://ai-gateway.vercel.sh/v1',
            check_embedding_ctx_length=False,  
            # **kwargs # TODO:: need to check 
        )


def init_chat_model(model: str, api_key: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Convenience function to initialize LLM via gateway"""
    return VercelGatewayLLM.create(model=model, api_key=api_key, **kwargs)


def init_embeddings(model: str, api_key: Optional[str] = None, **kwargs) -> Embeddings:
    """Convenience function to initialize embeddings via gateway"""
    return VercelGatewayEmbeddings.create(model=model, api_key=api_key, **kwargs)


