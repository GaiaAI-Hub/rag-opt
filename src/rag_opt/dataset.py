from typing_extensions import Doc
from rag_opt.llm import RAGLLM
from pydantic import BaseModel
# dataset generator 

# it should create qas 
# it should prepare datasets with contexts 

class RAGDatasetItem(BaseModel):
    question: str
    answer: str 
    contexts: list[str]

class RAGDataset(BaseModel):
    items: list[RAGDatasetItem]

    def to_json(self, path:str="./rag_dataset.json"):
        pass

    def from_json(self):
        pass 

    def from_dict(self, config_dict:dict):
        """Create RAGConfig from a dictionary"""
        # TODO:: evaluate the config
        return self(**config_dict)
    
class DatasetGenerator:
    llm: RAGLLM = None 
    # TODO:: make it compatible with HF Datasets
    def __init__(self, llm:RAGLLM=None):
        self.llm = llm

    def generate(self, n:Doc[int, "Number of QAs to generate"]=10) -> RAGDataset:
        pass

  