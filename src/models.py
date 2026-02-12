# models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List

class DocstringRequest(BaseModel):
    """Request model for docstring generation."""
    code: str = Field(
        ...,
        min_length=5,
        description="Python code (function, class, or method) to generate a docstring for",
        example="def add(a: int, b: int) -> int:\n    return a + b"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "def multiply(x, y):\n    return x * y"
            }
        }
    )


class DocstringResponse(BaseModel):
    """Response model for docstring generation."""
    success: bool = Field(..., description="Whether the generation was successful")
    docstring: Optional[str] = Field(None, description="Generated Google-style docstring")
    element_name: str = Field("unknown", description="Name of the function/class/method")
    element_type: Literal["function", "class", "method", "unknown"] = Field(
        "unknown", description="Type of Python element documented"
    )
    error: Optional[str] = Field(None, description="Error message if success is false")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "docstring": '"""Multiply two numbers together.\n\nArgs:\n    x (int/float): First number\n    y (int/float): Second number\n\nReturns:\n    int/float: Product of x and y\n"""',
                "element_name": "multiply",
                "element_type": "function"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: Literal["healthy", "unhealthy"]
    service: str
    version: str
    supports: List[str]
    model: str
    agents: Optional[List[str]] = None
    safety_limits: Optional[dict] = None
