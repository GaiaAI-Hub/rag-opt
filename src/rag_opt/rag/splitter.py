from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from typing import Literal 
from langchain.schema import Document
from abc import ABC, abstractmethod

# HP: splitting_type, chunk_size, chunk_overlap
class BaseSplitter(ABC):
    """ Base Splitter Class provides 3 levels of splitting for our RAG pipeline
        based on this 
        https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

        You can extend it for your own needs
    """
    def __init__(self,
                 splitting_type: Literal["characters", "recursive_character", "tokens"],
                 chunk_size:int,
                 chunk_overlap:int=5,
                ) -> None:
                self.splitting_type = splitting_type
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self,docs:list[Document]) -> list[Document]:
        if self.splitting_type == "characters":
            return self._split_characters(docs)
        elif self.splitting_type == "recursive_character":
            return self._split_recursive_character(docs)
        elif self.splitting_type == "tokens":
            return self._split_tokens(docs)
    
    def _split_characters(self,docs:list[Document]):
        text_splitter = CharacterTextSplitter(chunk_size = self.chunk_size, 
                                              chunk_overlap=self.chunk_overlap, 
                                              separator="\n\n",
                                              strip_whitespace=False)
        return text_splitter.split_documents(self.docs)

    def _split_recursive_character(self,docs:list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, 
                                                        chunk_overlap=self.chunk_overlap, 
                                                        length_function=len)
        return text_splitter.split_documents(self.docs)
    
    def _split_tokens(self,docs:list[Document]):
        text_splitter = TokenTextSplitter(chunk_size = self.chunk_size, 
                                          chunk_overlap=self.chunk_overlap,
                                          encoding_name="cl100k_base")
        return text_splitter.split_documents(self.docs)

    def _split_sementic(self,docs:list[Document]):
        raise NotImplementedError()
    
    def _split_agentic(self,docs:list[Document]):
        raise NotImplementedError()

class Splitter(BaseSplitter):
    """ Splitter Class  for our RAG pipeline (Phase 2)
    """
    def __init__(self,
                 splitting_type: Literal["characters", "recursive_character", "tokens"],
                 chunk_size:int,
                 chunk_overlap:int=5,
                ) -> None:
        """ Load the parsed documents and split them

            Args: (those args represents hyperparameters in our RAG pipeline)
            splitting_type: Type of splitting criteria to use.
            chunk_size: Size of each chunk.
            chunk_overlap: Overlap between chunks.
        """
        super().__init__(splitting_type, chunk_size, chunk_overlap)
    
    def split(self,docs:list[Document]) -> list[Document]:
        return super().split(docs)