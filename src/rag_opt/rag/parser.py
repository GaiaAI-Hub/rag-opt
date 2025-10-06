from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
                                            CSVLoader,
                                            UnstructuredExcelLoader,
                                            PyPDFLoader,
                                            # UnstructuredFileLoader,
                                            TextLoader,
                                            DirectoryLoader,
                                            DataFrameLoader
                                        )
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import TYPE_CHECKING,  Any, Optional,  Iterator
import tempfile
import logging
import shutil
import os
import pandas as pd
from pathlib import Path
from loguru import logger

if TYPE_CHECKING:
    from pandas import DataFrame

logging.getLogger("pdfminer").setLevel(logging.ERROR)

class CustomFileLoader(BaseLoader):
    """Custom loader that uses appropriate loader based on file type."""
    
    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        
    def lazy_load(self) -> Iterator[Document]:
        file_path = Path(self.file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            loader = CSVLoader(
                file_path=str(file_path),
                **self.kwargs
            )
        elif file_extension == '.xlsx' or file_extension == '.xls':
            loader = UnstructuredExcelLoader(
                file_path=str(file_path),
                mode="single",
                **self.kwargs
            )
        elif file_extension == '.pdf':
            loader = PyPDFLoader(
                str(file_path),
                **self.kwargs
            )
        elif file_extension == '.txt':
            loader = TextLoader(
                str(file_path),
                **self.kwargs
            )
        else:
            loader = UnstructuredLoader(
                str(file_path),
                **self.kwargs
            )
        try:
            for doc in loader.lazy_load():
                yield doc
        except NotImplementedError:
            # Fallback if lazy_load not implemented
            for doc in loader.load():
                yield doc
            
class Parser:
    """Enhanced Parser with custom loader support.
    This represents Phase 1 in RAG pipeline.
    """
    
    def __init__(self, 
                 path: str,
                 glob: str = "**/[!.]*",
                 include_sub_dir: bool = False,
                 silent_errors: bool = True,
                 use_multithreading: bool = True,
                 max_concurrency: int = 4,
                 csv_loader_kwargs: Optional[dict[str, Any]] = None,
                 excel_loader_kwargs: Optional[dict[str, Any]] = None,
                 **kwargs):
        """Initialize parser with path to data and custom loader options.
        
        Args: 
            path: Path to directory or file.
            glob: A glob pattern or list of glob patterns to use to find files.
            csv_loader_kwargs: Keyword arguments for CSVDocumentLoader
            excel_loader_kwargs: Keyword arguments for ExcelDocumentLoader
            **kwargs: Additional arguments for DirectoryLoader.
        """
        self._tmp_dir: str = None
        self.csv_loader_kwargs = csv_loader_kwargs or {}
        self.excel_loader_kwargs = excel_loader_kwargs or {}
        
        # Check if it is a file, create tmp dir and load 
        if not os.path.isdir(path):
            path = self._create_tmp_dir(path)
            self._tmp_dir = path
            
     
        self.loader = DirectoryLoader(
            path,
            glob=glob,
            silent_errors=silent_errors,
            show_progress=True,
            recursive=include_sub_dir,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
            loader_cls=CustomFileLoader,
            **kwargs
        )
    
    def _create_tmp_dir(self, file_path: str) -> str:
        """Create temporary directory in case user provides a file not directory."""
        tmp_dir = tempfile.mkdtemp()  
        basename = os.path.basename(file_path)
        tmp_file_path = os.path.join(tmp_dir, basename)
        shutil.copy(file_path, tmp_file_path)
        return tmp_dir
    
    def load_text(self) -> list[str]:
        """Load the data folder files into list of strings."""
        docs = self.load_docs()
        logger.debug(f"Retrieved Contexts: {docs[0]}")
        raw_texts = [doc.page_content for doc in docs]
        return raw_texts
    
    def load_docs(self) -> list[Document]:
        """Load the data folder files into langchain documents."""
        docs = self.loader.load()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir)
            self._tmp_dir = None
        return docs 
    
    @staticmethod
    def from_df(df: "DataFrame", 
                page_content_column: str = None,
                ) -> list[Document]:
        """Load data from pandas dataframe with enhanced metadata support."""
        _drop = False
        if page_content_column is None:
            _drop = True
            page_content_column = "gaia_all_text"
            df[page_content_column] = df.apply(
                lambda row: " | ".join(f"{col}: {row[col]}" for col in df.columns), 
                axis=1
            )
        
        loader = DataFrameLoader(df, page_content_column=page_content_column)
        docs = loader.load()
        
        if _drop and "gaia_all_text" in df.columns:
            df = df.drop(columns=["gaia_all_text"])
            
        return docs
