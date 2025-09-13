


from langchain_community.document_loaders import DirectoryLoader, DataFrameLoader
from langchain.schema import Document
from typing import TYPE_CHECKING
import tempfile
import logging
import shutil
import os 
if TYPE_CHECKING:
    from pandas import DataFrame

logging.getLogger("pdfminer").setLevel(logging.ERROR)


class Parser:
    """ parse raw data and create langchain documents ready to get splitted embedded 
        This represent Phase 1 in RAG pipeline
    """

    _tmp_dir:str = None

    def __init__(self, 
                 path:str,
                 glob:str="**/[!.]*",
                 include_sub_dir:bool=False,
                 silent_errors:bool=True,
                 use_multithreading:bool=True,
                 max_concurrency:int=4,
                 **kwargs ):
        """ initialize parser with path to data 
        Args: 
            path: Path to directory.
            glob: A glob pattern or list of glob patterns to use to find files.
                Defaults to "**/[!.]*" (all files except hidden).
            silent_errors: Whether to silently ignore errors. Defaults to True.
            load_hidden: Whether to load hidden files. Defaults to False.
            include_sub_dir: Whether to include subdirectories. Defaults to False.
            use_multithreading: Whether to use multithreading. Defaults to True.
            max_concurrency: The maximum number of threads to use. Defaults to 4.

            for other Args to customize the load pls check
            # https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/document_loaders/directory.py#L29

        """
        # TODO:: check if it is a file create tmp dir and load 
        if not os.path.isdir(path):
            path = self._create_tmp_dir(path)
            self._tmp_dir = path

        self.loader = DirectoryLoader(path,
                                      glob=glob,
                                      silent_errors=silent_errors,
                                      show_progress=True,
                                      recursive=include_sub_dir,
                                      use_multithreading=use_multithreading,
                                      max_concurrency=max_concurrency,
                                      **kwargs
                                      )
    
    def _create_tmp_dir(self,file_path:str):
        """ create temporary directory in case user provides a file not directory """
        tmp_dir = tempfile.mkdtemp()  
        basename = os.path.basename(file_path)
        tmp_file_path = os.path.join(tmp_dir, basename)
        shutil.copy(file_path, tmp_file_path)
        return tmp_dir
    
    def load_text(self) -> list[str]:
        """ load the data folder files into list of string """
        docs = self.load_docs()
        raw_text = [doc.page_content for doc in docs]
        return  raw_text
    
    def load_docs(self) -> list[Document]:
        """ load the data folder files into langchain documents """
        docs = self.loader.load()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir)
        return docs 
    
    @staticmethod
    def from_df(df:"DataFrame",page_content_column:str=None) -> list[Document]:
        """ load data from pandas dataframe """
        if page_content_column is None:
            _drop = True
            page_content_column = "gaia_all_text"
            df[page_content_column] = df.apply(lambda row: " | ".join(f"{col}: {row[col]}" for col in df.columns), axis=1)
        
        loader = DataFrameLoader(df, page_content_column=page_content_column)
        if _drop and "gaia_all_text" in df.columns:
            df = df.drop(columns=["gaia_all_text"])
        return loader.load()

