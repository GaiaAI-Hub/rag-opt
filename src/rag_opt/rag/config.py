from dataclasses import dataclass 


# TODO:: custom settings class based on pydantic or whatso ever 
# to load configs from yml, json , inline 
# allow apikeys , names ,. ..
# TODO:: allow custom config
@dataclass
class RAGConfig:
    """Configuration class for RAG hyperparameters"""
    
    # indexing
    chunk_size: int = 1000
    chunk_overlap: int = 50
    vector_store_provider: str = "faiss"
    search_type: str = "similarity"
    
    # generation
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    k: int = 4 # for RAG 

    # Embedding 
    embedding_provider: str = "openai"

    # reranking 
    use_reranker: bool = False
    reranker_type: str = "cross_encoder"

    @classmethod
    def from_dict(cls, config_dict):
        """Create RAGConfig from a dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Create RAGConfig from a YAML file"""

    def to_json(self,path:str="./rag_config.json"):
        pass