# agents.py
import ast
import textwrap
import re
import hashlib
import asyncio
from typing import Tuple, Dict, Any, Optional, Union
from enum import Enum
from functools import lru_cache
from dataclasses import dataclass
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseLLM

from app.config import get_llm, settings


# ============ Custom Exceptions ============

class DocstringGenerationError(Exception):
    """Base exception for docstring generation failures."""
    pass

class CodeValidationError(DocstringGenerationError):
    """Raised when code validation fails."""
    pass

class LLMGenerationError(DocstringGenerationError):
    """Raised when LLM call fails."""
    pass


# ============ Enums and Data Classes ============

class ElementType(str, Enum):
    """Types of Python elements that can be documented."""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    STATIC_METHOD = "staticmethod"
    CLASS_METHOD = "classmethod"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"


@dataclass
class DocstringResult:
    """Structured result of docstring generation."""
    docstring: str
    element_name: str
    element_type: ElementType
    language: str = "python"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "docstring": self.docstring,
            "element_name": self.element_name,
            "element_type": self.element_type.value
        }


# ============ Enhanced Validation with AST Analysis ============

def validate_and_analyze(code: str) -> Tuple[str, ElementType, str, Optional[Dict[str, Any]]]:
    """
    Validates Python code and extracts comprehensive metadata.
    
    Args:
        code: Raw Python code string
    
    Returns:
        Tuple of (cleaned_code, element_type, element_name, additional_metadata)
    
    Raises:
        CodeValidationError: If code is invalid or contains no documentable elements
    """
    if not isinstance(code, str) or not code.strip():
        raise CodeValidationError("Input must be a non-empty string")
    
    # Handle pasted methods/blocks with leading indentation
    cleaned = textwrap.dedent(code.strip())
    
    # Check for empty code after dedent
    if not cleaned or cleaned.isspace():
        raise CodeValidationError("Code contains only whitespace")
    
    try:
        parsed = ast.parse(cleaned)
    except SyntaxError as e:
        raise CodeValidationError(f"Invalid Python syntax: {e}")
    except IndentationError as e:
        raise CodeValidationError(f"Invalid indentation: {e}")
    
    if not parsed.body:
        raise CodeValidationError("No valid Python statements found.")
    
    # Get the first primary node (we document one thing at a time)
    node = parsed.body[0]
    
    # Skip if it's just a docstring or expression
    if isinstance(node, (ast.Expr, ast.Constant)) and not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        raise CodeValidationError("No function, class, or method definition found. Found only an expression or docstring.")
    
    metadata = {}
    
    # Class detection
    if isinstance(node, ast.ClassDef):
        # Check for decorators
        decorators = [d.id if isinstance(d, ast.Name) else None for d in node.decorator_list]
        metadata["decorators"] = [d for d in decorators if d]
        metadata["base_classes"] = [base.id if isinstance(base, ast.Name) else type(base).__name__ 
                                   for base in node.bases]
        return cleaned, ElementType.CLASS, node.name, metadata
    
    # Function/Method detection with enhanced classification
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Check decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
        
        metadata["decorators"] = decorators
        metadata["has_args"] = bool(node.args.args)
        metadata["has_return"] = bool(node.returns)
        metadata["is_async"] = is_async
        
        # Determine exact element type
        if 'staticmethod' in decorators:
            element_type = ElementType.STATIC_METHOD
        elif 'classmethod' in decorators:
            element_type = ElementType.CLASS_METHOD
        elif 'property' in decorators:
            element_type = ElementType.PROPERTY
        else:
            # Heuristic: if 'self' or 'cls' is the first arg, it's likely a method
            is_method = False
            if node.args.args:
                first_arg = node.args.args[0].arg
                is_method = first_arg in ['self', 'cls']
            
            if is_method:
                element_type = ElementType.ASYNC_METHOD if is_async else ElementType.METHOD
            else:
                element_type = ElementType.ASYNC_FUNCTION if is_async else ElementType.FUNCTION
        
        return cleaned, element_type, node.name, metadata
    
    else:
        raise CodeValidationError(
            f"No Python function, class, or method definition found. Found: {type(node).__name__}"
        )


# ============ PROMPTS - With FULL, DETAILED Examples ============

def create_function_prompt(code: str):
    """Create prompt for function docstring generation with FULL example."""
    messages = [
        SystemMessage(content="""You are a Python documentation expert. Generate COMPLETE Google-style docstrings.

CRITICAL REQUIREMENTS:
1. Return ONLY the raw docstring with triple quotes (\"\"\")
2. Include ALL sections that apply (Args, Returns, Raises, Yields, Note, Example)
3. Be thorough - at least 3-5 lines of description
4. Include parameter types and descriptions
5. Include return type and description
6. Document exceptions if any are raised
7. Add a simple example if the usage is non-obvious

EXAMPLE - FUNCTION (STUDY THIS CAREFULLY):
----------------------------------------------------------------
Code:
def calculate_statistics(numbers, ignore_zeros=False, precision=2):
    \"\"\"
    Calculate comprehensive statistical measures from a list of numbers.
    
    This function computes mean, median, mode, standard deviation, variance,
    min, max, and range. It handles edge cases like empty lists, single values,
    and optionally ignores zero values for specialized calculations.
    
    Args:
        numbers (list of float/int): Input numerical values. Must contain at least
            one non-zero element if ignore_zeros=True.
        ignore_zeros (bool, optional): If True, filter out zero values before
            calculations. Useful for datasets where zeros represent missing data.
            Defaults to False.
        precision (int, optional): Number of decimal places to round results to.
            Must be between 0 and 10. Defaults to 2.
    
    Returns:
        dict: A dictionary containing the following statistical measures:
            - 'mean' (float): Arithmetic mean
            - 'median' (float): Middle value (or average of two middle values)
            - 'mode' (float or None): Most frequent value, None if no unique mode
            - 'std' (float): Sample standard deviation
            - 'variance' (float): Sample variance
            - 'min' (float): Minimum value
            - 'max' (float): Maximum value
            - 'range' (float): Difference between max and min
            - 'count' (int): Number of values used in calculations
    
    Raises:
        ValueError: If input list is empty, contains non-numeric values, or
            if all values are zero when ignore_zeros=True.
        TypeError: If precision is not an integer or outside valid range.
    
    Example:
        >>> stats = calculate_statistics([1, 2, 2, 3, 4], precision=3)
        >>> print(stats['mean'])
        2.4
        >>> print(stats['mode'])
        2.0
    \"\"\"
    # Function implementation...

Now generate a SIMILARLY COMPREHENSIVE docstring for this function:"""),
        HumanMessage(content=code)
    ]
    return ChatPromptTemplate.from_messages(messages)

def create_class_prompt(code: str):
    """Create prompt for class docstring generation with FULL example."""
    messages = [
        SystemMessage(content="""You are a Python documentation expert. Generate COMPLETE Google-style class docstrings.

CRITICAL REQUIREMENTS:
1. Return ONLY the raw docstring with triple quotes (\"\"\")
2. Start with a clear, detailed class description (3-5 sentences)
3. Include Attributes: section with ALL instance/class variables
4. Include Methods: section summarizing key methods
5. Include Example: section showing typical usage

EXAMPLE - CLASS (STUDY THIS CAREFULLY):
----------------------------------------------------------------
Code:
class DataPipeline:
    def __init__(self, source, transformers=None, validate=True):
        self.source = source
        self.transformers = transformers or []
        self.validate = validate
        self.data = None
        self.metadata = {}
        self._processed_count = 0
    
    def load(self):
        pass
    
    def transform(self):
        pass
    
    def save(self, destination):
        pass

Docstring:
\"\"\"
A flexible data processing pipeline with pluggable transformation stages.

The DataPipeline class orchestrates ETL (Extract, Transform, Load) operations
with comprehensive validation, error handling, and performance monitoring.
It supports chaining multiple transformers, lazy loading, and partial
execution for large datasets. The pipeline maintains execution metadata
and can resume from failure points.

Attributes:
    source (str or Path): Data source location. Can be file path, URL,
        or database connection string.
    transformers (list): Ordered list of transformer objects. Each transformer
        must implement transform(data) method. Empty by default.
    validate (bool): If True, performs schema validation after each stage.
        Defaults to True.
    data (any): Currently loaded/transformed data. Initially None until
        load() or transform() is called.
    metadata (dict): Execution statistics including timestamps, row counts,
        and error logs. Populated during pipeline execution.
    _processed_count (int): Internal counter for processed records.
        For internal use only.

Methods:
    load(): Loads raw data from the configured source.
    transform(): Applies all transformer stages sequentially.
    save(destination): Writes processed data to specified location.
    validate_schema(): Validates data structure (internal method).
    get_stats(): Returns execution performance metrics.

Example:
    >>> pipeline = DataPipeline('data/raw.csv', 
    ...                        transformers=[Cleaner(), Normalizer()])
    >>> pipeline.load()
    >>> pipeline.transform()
    >>> pipeline.save('data/processed.parquet')
    >>> print(pipeline.metadata['row_count'])
    15423

Note:
    This class is thread-safe and implements context manager protocol
    for automatic resource cleanup.
\"\"\"

Now generate a SIMILARLY COMPREHENSIVE docstring for this class:"""),
        HumanMessage(content=code)
    ]
    return ChatPromptTemplate.from_messages(messages)

def create_method_prompt(code: str):
    """Create prompt for method docstring generation with FULL example."""
    messages = [
        SystemMessage(content="""You are a Python documentation expert. Generate COMPLETE Google-style method docstrings.

CRITICAL REQUIREMENTS:
1. Return ONLY the raw docstring with triple quotes (\"\"\")
2. EXCLUDE 'self' or 'cls' from Args section
3. Describe the method's role in the class's behavior
4. Include detailed Args, Returns, Raises
5. Include side effects or state changes

EXAMPLE - METHOD (STUDY THIS CAREFULLY):
----------------------------------------------------------------
Code:
    def process_batch(self, items, parallel=True, timeout=30, retry=3):
        # Method implementation...

Docstring:
\"\"\"
Process a batch of items through the pipeline with parallel execution.

This method distributes the workload across multiple worker threads,
significantly improving throughput for large batches. It implements
automatic retry logic for failed items and collects detailed metrics
about processing success rates and timing.

The method updates internal state including:
- self.processed_count: Incremented for each successful item
- self.failed_items: Appends failed items with error details
- self.performance_log: Records timing for each batch

Args:
    items (list): Collection of items to process. Each item must be
        compatible with the pipeline's transform() method.
    parallel (bool, optional): Enable concurrent processing using
        ThreadPoolExecutor. Defaults to True.
    timeout (int, optional): Maximum seconds to wait for batch completion.
        Raises TimeoutError if exceeded. Defaults to 30.
    retry (int, optional): Number of retry attempts for failed items.
        Set to 0 to disable retries. Defaults to 3.

Returns:
    dict: Processing results containing:
        - 'success_count' (int): Number of successfully processed items
        - 'failure_count' (int): Number of items that failed all retries
        - 'total_time' (float): Total processing time in seconds
        - 'avg_time_per_item' (float): Average processing time
        - 'failures' (list): Items that failed with error details

Raises:
    ValueError: If items is empty or contains invalid elements.
    TimeoutError: If processing exceeds the specified timeout.
    ResourceWarning: If system resources are insufficient for parallel
        processing (logged, not raised).

Yields:
    None. This method processes the entire batch synchronously and
    returns aggregated results. For streaming results, use process_stream().

Note:
    The batch processor maintains a thread pool that is reused across
    calls to minimize overhead. Call shutdown() to release resources.

Example:
    >>> processor = DataProcessor()
    >>> results = processor.process_batch(
    ...     items=range(1000),
    ...     parallel=True,
    ...     timeout=60,
    ...     retry=2
    ... )
    >>> print(f"Success rate: {results['success_count']/1000*100:.1f}%")
    97.3%
\"\"\"

Now generate a SIMILARLY COMPREHENSIVE docstring for this method:"""),
        HumanMessage(content=code)
    ]
    return ChatPromptTemplate.from_messages(messages)

def create_property_prompt(code: str):
    """Create prompt for property docstring generation with FULL example."""
    messages = [
        SystemMessage(content="""You are a Python documentation expert. Generate COMPLETE Google-style property docstrings.

CRITICAL REQUIREMENTS:
1. Return ONLY the raw docstring with triple quotes (\"\"\")
2. Describe WHAT is being computed/returned in detail
3. Include Returns: section with type and comprehensive description
4. Document any caching behavior or performance characteristics

EXAMPLE - PROPERTY (STUDY THIS CAREFULLY):
----------------------------------------------------------------
Code:
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

Docstring:
\"\"\"
Get the complete formatted name by combining first and last name.

This property provides a consistent representation of the entity's full name,
automatically handling edge cases such as missing name components, special
characters, and title prefixes. The result is memoized after first access
for performance.

The formatting follows these rules:
    - First and last names are capitalized appropriately
    - Multiple middle names are joined with spaces
    - Titles (Dr., Prof., etc.) are preserved if present
    - Unicode characters are normalized to NFC form
    - Extra whitespace is trimmed

Returns:
    str: The complete formatted name string. If first_name is missing,
        returns last_name only. If both are missing, returns "Unknown".
        Never returns None or empty string.

Caching:
    The formatted result is cached after first computation to avoid
    repeated string operations. Clear the cache by deleting the property:
    del instance.full_name

Example:
    >>> user = User(first_name="john", last_name="doe")
    >>> user.full_name
    'John Doe'
    >>> user.first_name = "jane"
    >>> user.full_name  # Auto-updates on attribute change
    'Jane Doe'

Note:
    This property triggers a database lookup for the name components
    if they haven't been loaded yet. Use prefetch_related() when
    accessing this property in bulk operations.
\"\"\"

Now generate a SIMILARLY COMPREHENSIVE docstring for this property:"""),
        HumanMessage(content=code)
    ]
    return ChatPromptTemplate.from_messages(messages)

def create_async_function_prompt(code: str):
    """Create prompt for async function docstring generation with FULL example."""
    messages = [
        SystemMessage(content="""You are a Python documentation expert. Generate COMPLETE Google-style docstrings for async functions.

CRITICAL REQUIREMENTS:
1. Return ONLY the raw docstring with triple quotes (\"\"\")
2. Explicitly mention async/await in the description
3. Document connection pooling and timeout behavior
4. Include error handling for network-related exceptions

EXAMPLE - ASYNC FUNCTION (STUDY THIS CAREFULLY):
----------------------------------------------------------------
Code:
async def fetch_api_data(endpoint, params=None, retry=3, timeout=10):
    # Implementation...

Docstring:
\"\"\"
Asynchronously fetch data from a REST API endpoint with automatic retry.

This coroutine implements a robust API client with connection pooling,
exponential backoff retry strategy, and comprehensive error handling.
It maintains a persistent session for multiple requests and automatically
handles rate limiting headers.

The function uses aiohttp for non-blocking HTTP requests and integrates
with asyncio's timeout mechanism to prevent hanging connections.

Args:
    endpoint (str): Full API URL or endpoint path. If relative path is
        provided, prepends the base URL from configuration.
    params (dict, optional): Query parameters to append to the request.
        Keys and values are automatically URL-encoded. Defaults to None.
    retry (int, optional): Maximum number of retry attempts for failed
        requests. Failed attempts use exponential backoff (1s, 2s, 4s).
        Defaults to 3. Set to 0 to disable retries.
    timeout (int, optional): Total timeout in seconds for the entire
        operation, including retries. Defaults to 10.

Returns:
    dict: Parsed JSON response from the API. Structure depends on the
        specific endpoint. Common fields include:
        - 'status' (str): 'success' or 'error'
        - 'data' (any): The requested resource
        - 'meta' (dict): Pagination and metadata
        - 'error' (dict): Error details if status is 'error'

Raises:
    aiohttp.ClientError: Base class for all aiohttp exceptions.
        Check specific subtypes for detailed error handling:
        - ClientConnectionError: Network connectivity issues
        - ClientResponseError: HTTP 4xx/5xx responses
        - ClientTimeoutError: Request exceeded timeout
    asyncio.TimeoutError: Operation exceeded total timeout including retries.
    ValueError: If endpoint is empty or invalid.

Yields:
    None. This is a regular coroutine that returns a single response.
    For streaming responses, use stream_api_data() instead.

Example:
    >>> data = await fetch_api_data(
    ...     endpoint='/users/123',
    ...     params={'include': 'posts,comments'},
    ...     retry=5,
    ...     timeout=30
    ... )
    >>> print(data['user']['name'])
    'John Doe'

Note:
    This function maintains an internal connection pool that is reused
    across calls. Call close_api_session() during application shutdown
    to properly release resources.
\"\"\"

Now generate a SIMILARLY COMPREHENSIVE docstring for this async function:"""),
        HumanMessage(content=code)
    ]
    return ChatPromptTemplate.from_messages(messages)


# ============ Robust Output Cleaning ============

def clean_llm_output(output: str) -> str:
    """
    Removes unwanted formatting and ensures proper docstring format.
    """
    if not output:
        raise LLMGenerationError("Empty response from LLM")
    
    # Remove markdown code blocks
    output = re.sub(r'^(`{3,})[a-zA-Z]*\n?', '', output, flags=re.MULTILINE)
    output = re.sub(r'\n?`{3,}$', '', output, flags=re.MULTILINE)
    output = output.strip()
    
    # Extract content between triple quotes
    triple_quote_pattern = r'"""[\s\S]*?"""'
    matches = re.findall(triple_quote_pattern, output)
    
    if matches:
        # Use the longest match
        output = max(matches, key=len)
    else:
        # Wrap the entire response
        output = '"""\n' + output.strip() + '\n"""'
    
    # Ensure we have substantial content
    content = output[3:-3].strip()
    if len(content) < 50:  # Too short
        raise LLMGenerationError("Generated docstring is too short")
    
    return output


def validate_docstring(docstring: str) -> bool:
    """Quick validation of generated docstring."""
    if not docstring or len(docstring) < 100:  # Minimum length for useful docstring
        return False
    if not (docstring.startswith('"""') and docstring.endswith('"""')):
        return False
    content = docstring[3:-3].strip()
    if len(content) < 50:
        return False
    return True


# ============ Retry Logic ============

def retry_on_failure(max_retries=2, initial_delay=1):
    """Simple retry decorator with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise LLMGenerationError(f"Failed after {max_retries} attempts: {last_exception}")
        return wrapper
    return decorator


# ============ Caching ============

def _get_code_hash(code: str) -> str:
    """Generate a stable hash for code caching."""
    normalized = textwrap.dedent(code.strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


@lru_cache(maxsize=128)
def _cached_generate(code_hash: str, code: str, element_type_value: str) -> Dict[str, Any]:
    """
    Cached version of docstring generation.
    """
    element_type = ElementType(element_type_value)
    
    # Select appropriate prompt creator
    prompt_creators = {
        ElementType.CLASS: create_class_prompt,
        ElementType.METHOD: create_method_prompt,
        ElementType.STATIC_METHOD: create_method_prompt,
        ElementType.CLASS_METHOD: create_method_prompt,
        ElementType.PROPERTY: create_property_prompt,
        ElementType.ASYNC_FUNCTION: create_async_function_prompt,
        ElementType.ASYNC_METHOD: create_method_prompt,
        ElementType.FUNCTION: create_function_prompt,
    }
    
    prompt_creator = prompt_creators.get(element_type, create_function_prompt)
    prompt = prompt_creator(code)
    
    # INCREASE TOKEN LIMIT - We need more tokens for comprehensive docstrings
    code_complexity = max(1, len(code) // 50)  # More tokens for longer code
    token_limit = max(800, min(2048, code_complexity * 150))  # Much higher token limit
    
    llm = get_llm(max_tokens=token_limit)
    chain = prompt | llm | StrOutputParser()
    
    @retry_on_failure()
    def invoke_llm():
        return chain.invoke({}, config={"timeout": 45})  # Longer timeout
    
    response = invoke_llm()
    
    # Clean and format the docstring
    docstring = clean_llm_output(response)
    
    return {
        "docstring": docstring,
        "element_name": "",
        "element_type": element_type
    }


# ============ Main Public API ============

def generate_docstring(raw_code: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Orchestrates validation, prompt selection, and docstring generation.
    """
    try:
        # Validate and analyze code
        validated_code, element_type, element_name, metadata = validate_and_analyze(raw_code)
        
        # Generate docstring (cached or fresh)
        if use_cache:
            code_hash = _get_code_hash(validated_code)
            result = _cached_generate(code_hash, validated_code, element_type.value)
            result["element_name"] = element_name
            result["element_type"] = element_type.value
        else:
            # Select appropriate prompt creator
            prompt_creators = {
                ElementType.CLASS: create_class_prompt,
                ElementType.METHOD: create_method_prompt,
                ElementType.STATIC_METHOD: create_method_prompt,
                ElementType.CLASS_METHOD: create_method_prompt,
                ElementType.PROPERTY: create_property_prompt,
                ElementType.ASYNC_FUNCTION: create_async_function_prompt,
                ElementType.ASYNC_METHOD: create_method_prompt,
                ElementType.FUNCTION: create_function_prompt,
            }
            
            prompt_creator = prompt_creators.get(element_type, create_function_prompt)
            prompt = prompt_creator(validated_code)
            
            # HIGH token limit for comprehensive docstrings
            code_complexity = max(1, len(validated_code) // 50)
            token_limit = max(800, min(2048, code_complexity * 150))
            
            llm = get_llm(max_tokens=token_limit)
            chain = prompt | llm | StrOutputParser()
            
            @retry_on_failure()
            def invoke_llm():
                return chain.invoke({}, config={"timeout": 45})
            
            response = invoke_llm()
            docstring = clean_llm_output(response)
            
            if not validate_docstring(docstring):
                raise LLMGenerationError("Generated docstring failed validation - too short or incomplete")
            
            result = {
                "docstring": docstring,
                "element_name": element_name,
                "element_type": element_type.value
            }
        
        return result
        
    except CodeValidationError:
        raise
    except LLMGenerationError:
        raise
    except Exception as e:
        raise DocstringGenerationError(f"Failed to generate docstring: {str(e)}")


# ============ Async Version ============

async def generate_docstring_async(raw_code: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Async version of generate_docstring for use in async applications.
    """
    try:
        # Validate and analyze code
        validated_code, element_type, element_name, metadata = validate_and_analyze(raw_code)
        
        # Select appropriate prompt creator
        prompt_creators = {
            ElementType.CLASS: create_class_prompt,
            ElementType.METHOD: create_method_prompt,
            ElementType.STATIC_METHOD: create_method_prompt,
            ElementType.CLASS_METHOD: create_method_prompt,
            ElementType.PROPERTY: create_property_prompt,
            ElementType.ASYNC_FUNCTION: create_async_function_prompt,
            ElementType.ASYNC_METHOD: create_method_prompt,
            ElementType.FUNCTION: create_function_prompt,
        }
        
        prompt_creator = prompt_creators.get(element_type, create_function_prompt)
        prompt = prompt_creator(validated_code)
        
        # HIGH token limit
        code_complexity = max(1, len(validated_code) // 50)
        token_limit = max(800, min(2048, code_complexity * 150))
        
        llm = get_llm(max_tokens=token_limit)
        chain = prompt | llm | StrOutputParser()
        
        last_exception = None
        for attempt in range(2):
            try:
                response = await chain.ainvoke({}, config={"timeout": 45})
                break
            except Exception as e:
                last_exception = e
                if attempt < 1:
                    await asyncio.sleep(1)
                else:
                    raise LLMGenerationError(f"Async generation failed: {last_exception}")
        
        docstring = clean_llm_output(response)
        
        if not validate_docstring(docstring):
            return generate_docstring(raw_code, use_cache)
        
        return {
            "docstring": docstring,
            "element_name": element_name,
            "element_type": element_type.value
        }
        
    except CodeValidationError:
        raise
    except Exception as e:
        raise DocstringGenerationError(f"Failed to generate docstring async: {str(e)}")