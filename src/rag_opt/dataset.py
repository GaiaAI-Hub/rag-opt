from rag_opt._prompts import RAG_DATASET_GENERATION_PROMPT
from langchain.schema import Document, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain.schema import BasePromptTemplate
from typing_extensions import Doc, Annotated
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field
from rag_opt.rag import Parser
from rag_opt.llm import RAGLLM
from loguru import logger 
import asyncio
import random
import json


DIFFICULTY_CONFIGS = {
    "easy": {
        "description": "Generate basic recall questions that can be answered directly from the text",
        "sample_questions": ["What is X?", "Where does Y happen?", "When did Z occur?"]
    },
    "medium": {
        "description": "Generate analytical questions requiring understanding and synthesis",
        "sample_questions": ["How does X relate to Y?", "What are the implications of Z?", "Compare X and Y"]
    },
    "hard": {
        "description": "Generate complex questions requiring deep reasoning and inference",
        "sample_questions": ["Why might X lead to Y?", "What would happen if Z changed?", "Evaluate the effectiveness of X"]
    }
}

QuestionDifficulty = Literal["easy", "medium", "hard"]


class RAGDatasetItem(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

class RAGDataset(BaseModel):
    items: list[RAGDatasetItem]
    
    def to_json(self, path: str = "./rag_dataset.json") -> None:
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def from_json(cls, path: str) -> 'RAGDataset':
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())

class DatasetGenerator:
    _prompt: BasePromptTemplate = None 
    def __init__(self, 
                 llm: Annotated[RAGLLM, Doc("the llm to be used in the dataset generation process")],
                 *,
                 dataset_path: Annotated[str, Doc("the path to the dataset folder or file to be used by parser")] = None,
                 custom_prompt: Annotated[str, Doc("the custom prompt template to be used for question-answer generation")] = None,
                 parser: Annotated[Parser, Doc("the dataset loader to load and parse dataset document from a specific folder or file")] = None,
                 ):
        
        self.llm = llm

        # parser
        if dataset_path and not parser:
            parser= Parser(path=dataset_path)
        
        self.parser = parser
        self.dataset_path = dataset_path
        
        # Default prompt template for question-answer generation
        # TODO:: validate custom prompt if given (context is required as input_types)
        self._prompt = PromptTemplate(
            template=custom_prompt or RAG_DATASET_GENERATION_PROMPT,
            input_variables=["contexts", "difficulty_instruction"]
        )

    def _load_documents(self) -> list[Document]:
        """Load documents using parser if dataset_path is provided"""
        if not self.parser or not self.dataset_path:
            return []
        try:
            return self.parser.load_docs()
        except Exception as e:
            logger.error(f"Failed to load documents from {self.dataset_path}: {e}")
            return []

    def _get_difficulty_instruction(self, difficulty: QuestionDifficulty) -> str:
        """Get difficulty-specific instruction for prompt"""
        config = DIFFICULTY_CONFIGS[difficulty]
        return f"{config['description']}. Examples: {', '.join(config['sample_questions'])}"
    
    def _create_batch_prompts_with_contexts(self, contexts_batch: list[str], difficulties: list[QuestionDifficulty]) -> list[tuple[str, list[str]]]:
        """Create prompts for a batch of contexts with varying difficulties"""
        prompts_contexts = []
        for contexts_subset, difficulty in zip(self._chunk_contexts(contexts_batch, len(difficulties)), difficulties):
            difficulty_instruction = self._get_difficulty_instruction(difficulty)
            contexts= random.sample(contexts_subset, min(3,len(contexts_subset)))
            prompt = self._prompt.format(
                contexts=contexts,
                difficulty_instruction=difficulty_instruction
            )
            prompts_contexts.append((prompt, contexts))
        return prompts_contexts
    
    def _chunk_contexts(self, contexts: list[str], num_chunks: int) -> list[list[str]]:
        """Split contexts into chunks"""
        chunk_size = max(1, len(contexts) // max(1, num_chunks))
        return [contexts[i:i + chunk_size] for i in range(0, len(contexts), chunk_size)]
    
    def _parse_llm_responses(self, 
                             responses: list[BaseMessage], 
                             difficulties: list[QuestionDifficulty],
                             used_contexts: list[list[str]]) -> list[RAGDatasetItem]:
        """Parse LLM responses into dataset items"""
        items = []
        for response, difficulty, contexts in zip(responses, difficulties, used_contexts):
            try:
                data = json.loads(response.content if hasattr(response, 'content') else str(response))
                if isinstance(data, list):
                    # Multiple QA pairs
                    for qa in data:
                        items.append(self._create_dataset_item(qa, difficulty,contexts=contexts))
                else:
                    # Single QA pair
                    items.append(self._create_dataset_item(data, difficulty,contexts=contexts))
                    
            except json.JSONDecodeError:
                # Fallback to heuristic parsing
                fallback_item = self._parse_fallback_response(str(response), contexts, difficulty)
                if fallback_item:
                    items.append(fallback_item)
        return items

    def _create_dataset_item(self, qa_data: dict, 
                             difficulty: QuestionDifficulty,
                             contexts: list[str]) -> RAGDatasetItem:
        """Create RAGDatasetItem from parsed data"""
        logger.success(f"contexts: {contexts[0]}")
        return RAGDatasetItem(
            question=qa_data.get('question', ''),
            answer=qa_data.get('answer', ''),
            contexts=contexts,
            difficulty=difficulty
        )
        
    def _generate_batch_contexts_and_size(self,
                                 source_texts: Optional[list[str]] = None,
                                 source_docs: Optional[list[Document]] = None,
                                 n: int = 10,
                                 batch_size: Optional[int] = None,
                                 ) -> tuple[list[list[str]], int]:
        if not source_docs and not source_texts:
            # Try to load from parser if available
            if self.parser and self.dataset_path:
                logger.info(f"Loading documents from {self.dataset_path}")
                source_docs = self._load_documents()
                if not source_docs:
                    logger.error("Failed to load documents from parser")
                    raise ValueError("Failed to load documents from parser")
            else:
                logger.error("You have to provide at least one of source_texts or source_docs")
                raise ValueError("You have to provide at least one of source_texts or source_docs")

        # Convert docs to texts if needed
        if not source_texts:
            source_texts = [doc.page_content for doc in source_docs]
        
        # Filter out empty texts
        source_texts = [text.strip() for text in source_texts if text.strip()]
        if not source_texts:
            raise ValueError("No valid source texts found")

        # Set defaults
        batch_size = batch_size or max(min(n // 4, 1), 5, len(source_texts))
        logger.info(f"Generating {n} QA pairs with batch size {batch_size}")
        
        _batch_context_length = len(source_texts) // batch_size
        batches_contexts = [source_texts[i:i + _batch_context_length] for i in range(0, len(source_texts), _batch_context_length)]
        return batches_contexts, batch_size

    def _generate_difficulty_distribution(self, n: int) -> list[QuestionDifficulty]:
        """Generate balanced difficulty distribution"""
        # Default distribution: 30% easy, 50% medium, 20% hard
        easy_count = max(1, int(n * 0.3))
        hard_count = max(1, int(n * 0.2))
        medium_count = max(1,n - easy_count - hard_count)
        difficulties = (["easy"] * easy_count + 
                       ["medium"] * medium_count + 
                       ["hard"] * hard_count)
        
        random.shuffle(difficulties)
        return difficulties
    def _prepare_source_data(self, source_texts: Optional[list[str]], 
                           source_docs: Optional[list[Document]]) -> list[str]:
        """Prepare and validate source data"""
        if not source_docs and not source_texts:
            if self.parser and self.dataset_path:
                logger.info(f"Loading documents from {self.dataset_path}")
                source_docs = self._load_documents()
                if not source_docs:
                    raise ValueError("Failed to load documents from parser")
            else:
                raise ValueError("Must provide source_texts, source_docs, or dataset_path")

        # Convert docs to texts if needed
        if not source_texts and source_docs:
            source_texts = [doc.page_content for doc in source_docs]
        
        # Filter empty texts
        return [text.strip() for text in source_texts if text.strip()]
    def _parse_fallback_response(self, response: str, contexts: list[str], 
                               difficulty: QuestionDifficulty) -> Optional[RAGDatasetItem]:
        """Fallback parser for non-JSON responses"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            question = answer = ""
            
            for line in lines:
                lower_line = line.lower()
                if lower_line.startswith(('question:', 'q:')):
                    question = line.split(':', 1)[1].strip()
                elif lower_line.startswith(('answer:', 'a:')):
                    answer = line.split(':', 1)[1].strip()
            
            if question and answer:
                return RAGDatasetItem(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    difficulty=difficulty
                )
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
        
        return None
    
    def generate(self, 
                n: int = 10,
                 *,
                source_texts: Optional[list[str]] = None,
                source_docs: Optional[list[Document]] = None,
                batch_size: Optional[int] = None) -> RAGDataset:
        """Generate synthetic RAG dataset with question difficulty variance
        
        Args:
            source_texts: Source texts for question generation
            source_docs: Source documents for question generation  
            n: Number of questions to generate
            batch_size: Batch size for generation (default: adaptive)
            difficulty_distribution: Custom difficulty distribution
            
        Returns:
            RAGDataset: Generated dataset with varied question difficulties
        """
        # Prepare data
        texts = self._prepare_source_data(source_texts, source_docs)
        batch_size = batch_size or min(max(1, n // 4), 5)
        difficulties = self._generate_difficulty_distribution(n)

        logger.success(  f" {len(texts)} source texts prepared for generation")
        logger.success( f"{len(difficulties)} difficulties prepared for generation")
        
        # Generate in batches
        all_items = []
        text_batches = self._chunk_contexts(texts, batch_size)
        difficulty_batches = self._chunk_contexts(difficulties, len(text_batches))
        
        logger.info(f"Generating {len(text_batches)} batches of {batch_size} QA pairs")
        logger.info(f"Difficulty distribution: {difficulties}")
        try:
            for i, (text_batch, diff_batch) in enumerate(zip(text_batches, difficulty_batches)):
                logger.info(f"Generating batch {i+1}/{len(text_batches)}")
                
                prompts_contexts = self._create_batch_prompts_with_contexts(text_batch, diff_batch)
                prompts = [prompt for prompt, _ in prompts_contexts]
                contexts = [context for _, context in prompts_contexts]
                responses = self.llm.batch(prompts)
                
                batch_items = self._parse_llm_responses(responses, diff_batch, used_contexts=contexts)
                all_items.extend(batch_items)
                
        except KeyboardInterrupt:
            logger.warning("Generation interrupted by user")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
        
        logger.success(f"Generated {len(all_items)} QA pairs")
        return RAGDataset(items=all_items)
    
    async def agenerate(self,
                        n: int = 10,
                        *,
                        source_texts: Optional[list[str]] = None,
                        source_docs: Optional[list[Document]] = None,
                        batch_size: Optional[int] = None,
                        **kwargs):
        """ async generate datasets"""
        return await asyncio.to_thread(self.generate, **kwargs)
        # Prepare data
        # texts = self._prepare_source_data(source_texts, source_docs)
        # batch_size = batch_size or min(max(1, n // 4), 5)
        # difficulties = self._generate_difficulty_distribution(n)
        
        # # Generate in batches
        # all_items = []
        # text_batches = self._chunk_contexts(texts, batch_size)
        # difficulty_batches = self._chunk_contexts(difficulties, len(text_batches))
        # all_items: list[RAGDatasetItem] = []
        # tasks = []
        # try:
        #     for i, (text_batch, diff_batch) in enumerate(zip(text_batches, difficulty_batches)):
        #         logger.info(f"Generating batch {i+1}/{len(text_batches)}")
                
        #         prompts, contexts = self._create_batch_prompts_with_contexts(text_batch, diff_batch)
        #         task = asyncio.create_task(
        #             asyncio.to_thread(self.llm.abatch, prompts)
        #         )
        #         tasks.append(task)
            
        #     results = await asyncio.gather(*tasks, return_exceptions=True)
        #     for res in results:
        #         if isinstance(res, list):
        #             all_items.extend(res)
        #         elif isinstance(res, RAGDatasetItem):
        #             all_items.append(res)
        #         elif isinstance(res, Exception):
        #             logger.error(f"Async generation error: {res}")
      
        # except KeyboardInterrupt:
        #     logger.warning("Generation interrupted by user")
        # except Exception as e:
        #     logger.error(f"Error during generation: {e}")
        
        # logger.success(f"Successfully generated {len(all_items)} QA pairs")
        
        # return RAGDataset(items=all_items)
    