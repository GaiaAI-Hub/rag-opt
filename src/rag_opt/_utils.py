from langchain_core.prompts import get_template_variables
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from loguru import logger
import atexit
import re

_executor_registry: dict[int, ThreadPoolExecutor] = {}
_executor_lock = Lock()


def get_shared_executor(max_workers: int = 5) -> ThreadPoolExecutor:
    """
    Return a shared ThreadPoolExecutor with specified max_workers.
    
    FIXED: Properly handles different max_workers values by caching
    per worker count instead of ignoring the parameter.
    
    Args:
        max_workers: Maximum number of worker threads.
        
    Returns:
        ThreadPoolExecutor: Shared executor for this worker count.
    """
    with _executor_lock:
        if max_workers not in _executor_registry:
            logger.debug(f"Creating new shared executor with {max_workers} workers")
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"shared_pool_{max_workers}"
            )
            _executor_registry[max_workers] = executor
            
            # Register shutdown for this specific executor
            atexit.register(_shutdown_executor, max_workers)
        
        return _executor_registry[max_workers]


def _shutdown_executor(max_workers: int):
    """Shutdown specific executor at exit"""
    if max_workers in _executor_registry:
        logger.debug(f"Shutting down executor with {max_workers} workers")
        _executor_registry[max_workers].shutdown(wait=True)
        del _executor_registry[max_workers]


def get_dedicated_executor(max_workers: int = 5, name: str = "dedicated") -> ThreadPoolExecutor:
    """
    Create a new dedicated ThreadPoolExecutor (not shared/cached).
    
    Use this when you need isolated thread pools to avoid interference.
    Remember to call executor.shutdown() when done!
    
    Args:
        max_workers: Maximum number of worker threads.
        name: Name prefix for threads (for debugging).
        
    Returns:
        ThreadPoolExecutor: New dedicated executor instance.
    """
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=f"{name}_pool"
    )
    logger.debug(f"Created dedicated executor '{name}' with {max_workers} workers")
    return executor


def shutdown_all_executors():
    """Manually shutdown all shared executors (useful for testing)"""
    with _executor_lock:
        for max_workers, executor in list(_executor_registry.items()):
            logger.info(f"Shutting down executor with {max_workers} workers")
            executor.shutdown(wait=True)
        _executor_registry.clear()


def extract_num_from_text(text: str) -> int | None:
    """
    Fallback function to extract a number from text.
    
    Args:
        text: Input text potentially containing a number.
        
    Returns:
        Extracted integer or None if not found.
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


def validate_prompt(
    prompt_template: str, 
    reference_prompt: str, 
    raise_error: bool = True
) -> bool:
    """
    Validate that a prompt template contains all required variables.
    
    Args:
        prompt_template: User's prompt template to validate.
        reference_prompt: Reference prompt with required variables.
        raise_error: Whether to raise ValueError on validation failure.
        
    Returns:
        True if valid, False otherwise.
        
    Raises:
        ValueError: If raise_error=True and validation fails.
    """
    prompt_vars = get_template_variables(reference_prompt, "f-string")
    vars = get_template_variables(prompt_template, "f-string")
    
    needed_vars = set(prompt_vars) - set(vars)
    extra_vars = set(vars) - set(prompt_vars)
    
    # Check for missing variables
    if needed_vars:
        error_msg = f"Prompt missing required variables: {needed_vars}"
        logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False
    
    if extra_vars:
        warning_msg = (
            f"Prompt contains extra variables: {extra_vars}. "
            f"Required variables are: {prompt_vars}"
        )
        logger.warning(warning_msg)
    
    return True