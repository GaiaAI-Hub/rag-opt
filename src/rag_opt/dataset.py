from typing import Optional, Any, Annotated, TypeVar, Generic, Literal
from pydantic import BaseModel, Field
from langchain.schema import Document



QuestionDifficulty = Literal["easy", "medium", "hard"]

class GroundTruth(BaseModel):
    """ Generated from the train dataset item and used later for evaluation metric """
    answer: str
    contexts: list[str]

class ComponentUsage(BaseModel):
    """Base class for component-level metrics (cost, latency, tokens, etc.)"""
    llm: Annotated[float, Field(description="usage info of generation for one query like cost , latency, ...")]
    embedding: Annotated[float, Field(description="usage info of embedding for one query like cost , latency, ...")]
    vectorstore: Annotated[float, Field(description="usage info of vectorstore for one query like cost , latency, ...")]
    reranker: Annotated[float, Field(description="usage info of reranker for one query like cost , latency, ...")]

    @property
    def total(self) -> float:
        return self.llm + self.embedding + self.vectorstore + self.reranker


class TrainDatasetItem(BaseModel):
    question: str 
    answer: str 
    contexts: list[str] 
    difficulty: Annotated[Optional[QuestionDifficulty], Field(description="The difficulty level of the question", default=None)]
    def to_ground_truth(self) -> GroundTruth:
        """Convert training item to ground truth for evaluation"""
        return GroundTruth(answer=self.answer, contexts=self.contexts)
    
class EvaluationDatasetItem(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Annotated[GroundTruth, Field(description="The expected answer and contexts for the question")]
    metadata: Annotated[Optional[dict[str, Any]], Field(default_factory=dict, description="Generated information from callbacks for evaluation metrics")] 
    
    @property
    def cost(self) -> ComponentUsage:
        """ Cost metrics for one query """
        if "cost" in self.metadata:
            if isinstance(self.metadata["cost"], dict):
                return ComponentUsage(**self.metadata["cost"])
            return self.metadata["cost"]
        return ComponentUsage(llm=0, embedding=0, vectorstore=0, reranker=0)
    
    @property
    def latency(self) -> ComponentUsage:
        """ Latency metrics for one query """
        if "latency" in self.metadata:
            if isinstance(self.metadata["latency"], dict):
                return ComponentUsage(**self.metadata["latency"])
            return self.metadata["latency"]
        return ComponentUsage(llm=0, embedding=0, vectorstore=0, reranker=0)

    def set_cost(self, cost: ComponentUsage | dict) -> None:
        """Store cost metrics in metadata"""
        self.metadata["cost"] = cost.model_dump() if isinstance(cost, ComponentUsage) else cost
    
    def set_latency(self, latency: ComponentUsage | dict) -> None:
        """Store latency metrics in metadata"""
        self.metadata["latency"] = latency.model_dump() if isinstance(latency, ComponentUsage) else latency

T = TypeVar('T', bound=TrainDatasetItem|EvaluationDatasetItem)

class DatbasetMixin(BaseModel, Generic[T]):
    items: list[T]

    def to_langchain_docs(self) -> list[Document]:
        """Convert dataset items to langchain documents"""
        if self.items and isinstance(self.items[0], TrainDatasetItem): # train has no metadata 
            return [Document(page_content=f"Question: {d.question}\nAnswer: {d.answer}\nContexts: {d.contexts}") for d in self.items]
        
        return [Document(page_content=f"Question: {d.question}\nAnswer: {d.answer}", metadata=d.metadata) for d in self.items]

    def to_ground_truth(self) -> list[GroundTruth]:
        return [item.to_ground_truth() for item in self.items]
    
    def to_json(self, path: str = "./rag_dataset.json") -> None:
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def from_json(cls, path: str) -> T:
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())
    
    def __len__(self):
        return len(self.items)

class TrainDataset(DatbasetMixin[TrainDatasetItem]):
    """ dataset used in querying process"""
    items: list[TrainDatasetItem]
    

class EvaluationDataset(DatbasetMixin[EvaluationDatasetItem]):
    items: list[EvaluationDatasetItem]
    
    

