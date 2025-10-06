from langchain_core.prompts import get_template_variables
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import functools
import atexit
import re 


@functools.lru_cache(maxsize=1)
def get_shared_executor(max_workers:int=5) -> ThreadPoolExecutor:
    """ Return a lazily created, shared ThreadPoolExecutor.

    This ensures that only one executor instance is created and reused 
    throughout the lifetime of the program. The executor is automatically 
    shut down at program exit 

    Args:
        max_workers (int): Maximum number of worker threads for the executor.
                           Only applied on the first call.

    Returns:
        ThreadPoolExecutor: A shared executor instance.
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    atexit.register(executor.shutdown, wait=True)
    return executor


def extract_num_from_text(text: str) -> int | None:
    """
    Fallback function to extract a number from text
    """
    if not text:
        return None
    
    match = re.search(r"-?\d+", text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            logger.error(f"Failed to parse number from text: {text}")
            return None
    return None


def validate_prompt(prompt_template:str,reference_prompt:str, raise_error:bool=True) -> bool:
    prompt_vars = get_template_variables(reference_prompt, "f-string")
    vars = get_template_variables(prompt_template, "f-string")
    needed_vars = set(prompt_vars) - set(vars)
    diff = set(vars) ^ set(prompt_vars)
    if needed_vars:
        logger.error(f"Your prompt is missing the following variables: {needed_vars}")
        if raise_error:
            raise ValueError(f"Your prompt is missing the following variables: {needed_vars}")
        return False
    
    if diff:
        logger.error(f"Your prompt contains extra variables: {diff} \n your prompt must only contain the following variables only:  {prompt_vars}")
        
        if raise_error:
            raise ValueError(f"Your prompt contains extra variables: {diff}")
        return False
    
    return True 