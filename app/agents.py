# agents.py
import ast
import textwrap
import re
from typing import Tuple, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import get_llm, settings


def validate_and_analyze(code: str) -> Tuple[str, str, str]:
    """
    Validates Python code and extracts metadata with improved error handling.
    
    Args:
        code: Raw Python code string
    
    Returns:
        Tuple of (cleaned_code, element_type, element_name)
    
    Raises:
        ValueError: If code is invalid or contains no documentable elements
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Input must be a non-empty string")
    
    # Handle pasted methods/blocks with leading indentation
    cleaned = textwrap.dedent(code.strip())
    
    try:
        parsed = ast.parse(cleaned)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    except IndentationError as e:
        raise ValueError(f"Invalid indentation: {e}")
    
    if not parsed.body:
        raise ValueError("No valid Python statements found.")
    
    # Get the first primary node (we document one thing at a time)
    node = parsed.body[0]
    
    # Class detection
    if isinstance(node, ast.ClassDef):
        return cleaned, "class", node.name
    
    # Function/Method detection
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Heuristic: if 'self' or 'cls' is the first arg, it's likely a method
        is_method = False
        if node.args.args:
            first_arg = node.args.args[0].arg
            is_method = first_arg in ['self', 'cls']
        
        element_type = "method" if is_method else "function"
        return cleaned, element_type, node.name
    
    else:
        raise ValueError(
            f"No Python function, class, or method definition found. Found: {type(node).__name__}"
        )


# ============ Optimized Prompts - Static System Messages ============

BASE_SYSTEM_INSTRUCTIONS = """You are a Python documentation expert. Generate ONLY a Google-style docstring.

CRITICAL RULES:
- Do NOT wrap your response in ```python or ``` blocks
- Do NOT add any explanatory text before or after the docstring
- Return ONLY the raw docstring starting and ending with triple quotes (\"\"\")
- Use proper Google docstring format with clear sections

The docstring MUST follow this exact structure:
1. Brief description (one line)
2. Blank line
3. Args section (if applicable)
4. Blank line  
5. Returns section (if applicable)
6. Blank line
7. Raises section (if applicable)"""

# Specialized instructions for different element types
FUNCTION_INSTRUCTIONS = """For FUNCTIONS:
- Include Args: section with parameter names, types, and descriptions
- Include Returns: section with return type and description
- Include Raises: section if the function explicitly raises exceptions
- Be concise but thorough in the description"""

CLASS_INSTRUCTIONS = """For CLASSES:
- Start with a clear description of the class purpose
- Include Attributes: section listing instance/class variables
- Optionally include a Methods: section summarizing key methods
- Add an Example: section only if the usage pattern is non-obvious"""

METHOD_INSTRUCTIONS = """For METHODS:
- Contextualize it as a method of its parent class
- EXCLUDE 'self' or 'cls' from the Args section
- Describe the method's role in the class's behavior
- Include Args: for all other parameters
- Include Returns: section with type and description"""

# Compact prompts - static instructions in system message
function_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{BASE_SYSTEM_INSTRUCTIONS}\n\n{FUNCTION_INSTRUCTIONS}"),
    ("human", "Generate docstring for:\n\n{code}")
])

class_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{BASE_SYSTEM_INSTRUCTIONS}\n\n{CLASS_INSTRUCTIONS}"),
    ("human", "Generate docstring for:\n\n{code}")
])

method_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{BASE_SYSTEM_INSTRUCTIONS}\n\n{METHOD_INSTRUCTIONS}"),
    ("human", "Generate docstring for:\n\n{code}")
])


def clean_llm_output(output: str) -> str:
    """
    Removes unwanted formatting and ensures proper docstring format.
    Optimized for speed and reliability.
    """
    if not output:
        return '"""\n    TODO: Add docstring\n"""'
    
    # Fast path: if already properly formatted
    output = output.strip()
    
    # Remove markdown code blocks - simple regex
    output = re.sub(r'^`{3}(?:python)?\s*', '', output, flags=re.IGNORECASE)
    output = re.sub(r'\s*`{3}$', '', output)
    
    # Extract docstring if there's extra text
    if '"""' in output:
        # Find first and last triple quotes
        start = output.find('"""')
        end = output.rfind('"""')
        if start != -1 and end != -1 and end > start:
            output = output[start:end + 3]
    else:
        # Wrap in triple quotes
        output = '"""\n' + output + '\n"""'
    
    # Fix common formatting issues
    output = re.sub(r'"""\s*"""', '"""\n"""', output)  # Fix empty docstrings
    output = re.sub(r'\n{4,}', '\n\n', output)  # Limit consecutive newlines
    
    return output


def generate_docstring(raw_code: str) -> Dict[str, Any]:
    """
    Orchestrates validation, prompt selection, and docstring generation.
    Uses shared LLM instance with optimized token usage.
    
    Args:
        raw_code: Raw Python code string
    
    Returns:
        Dictionary containing docstring, element_name, and element_type
    
    Raises:
        ValueError: If validation or generation fails
    """
    try:
        # Validate and analyze code
        validated_code, element_type, element_name = validate_and_analyze(raw_code)
        
        # Select appropriate prompt
        prompt_map = {
            "class": class_prompt,
            "method": method_prompt,
            "function": function_prompt,
        }
        prompt = prompt_map.get(element_type, function_prompt)
        
        # Use shared LLM instance with reduced token limit
        llm = get_llm(max_tokens=settings.MAX_TOKENS_DOCSTRING)
        
        # Create chain and invoke with timeout
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"code": validated_code})
        
        # Clean and format the docstring
        docstring = clean_llm_output(response)
        
        return {
            "docstring": docstring,
            "element_name": element_name,
            "element_type": element_type
        }
        
    except ValueError:
        # Re-raise validation errors directly
        raise
    except Exception as e:
        raise ValueError(f"Failed to generate docstring: {str(e)}")