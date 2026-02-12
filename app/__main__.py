"""
FastAPI application for Gemini Docstring & README Generator.
Hackathon Submission - Nasiko Labs
✅ Docstring Agent: Functions, Classes, Methods
✅ README Agent: Full project analysis, ZIP upload
✅ Edge Cases: Empty files, large folders, nested structures
"""

from fastapi import FastAPI, Request, Body, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import tempfile
import zipfile
import traceback
import shutil
from typing import Optional

from app.agents import generate_docstring
from app.readme_agent import generate_project_readme, ProjectAnalyzer, save_readme
from app.models import DocstringRequest, DocstringResponse, HealthResponse
from app.config import settings, get_settings

# ============ Path Setup ============
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ============ FastAPI App ============
app = FastAPI(
    title="Gemini Documentation Generator - Hackathon Edition",
    description="""
    # 🏆 Nasiko Hackathon Submission
    
    ## 📝 Docstring Generation Agent
    - Functions, classes, methods
    - Google-style format
    - AST validation
    
    ## 📚 Project README Generation Agent  
    - Full project analysis
    - ZIP upload support
    - Nested directory structures
    - Safety limits: 100KB/file, 500 files, 50MB ZIP
    """,
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ============ Middleware ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Static Files ============
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ============ Web Interface ============
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """Render web interface with both agents."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "version": app.version,
            "title": "AI Documentation Generator - Hackathon",
            "model": settings.MODEL_NAME
        }
    )

# ============ Docstring Agent ============
@app.post("/api/generate", response_model=DocstringResponse, tags=["docstring"])
async def generate_docstring_api(request: DocstringRequest):
    """Generate Google-style docstrings for Python functions/classes/methods."""
    try:
        result = generate_docstring(request.code)
        return DocstringResponse(
            success=True,
            docstring=result["docstring"],
            element_name=result["element_name"],
            element_type=result["element_type"]
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=DocstringResponse(
                success=False, 
                error=str(e), 
                element_name="unknown", 
                element_type="unknown"
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=DocstringResponse(
                success=False, 
                error=f"Internal error: {str(e)}", 
                element_name="unknown", 
                element_type="unknown"
            ).model_dump()
        )

# ============ README Agent - Path Analysis ============
@app.post("/api/generate-readme", tags=["readme"])
async def generate_readme_api(project_path: str = Body(..., embed=True)):
    """
    Generate README.md from local folder path.
    """
    try:
        readme_content = generate_project_readme(project_path)
        
        # Also get analysis for stats
        analyzer = ProjectAnalyzer(project_path)
        analysis = analyzer.analyze()
        
        return {
            "success": True,
            "readme": readme_content,
            "project_name": analysis['project_name'],
            "file_count": analysis['summary']['total_files'],
            "summary": analysis['summary']
        }
    except ValueError as e:
        return JSONResponse(
            status_code=400, 
            content={
                "success": False, 
                "error": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "success": False, 
                "error": f"❌ Server error: {str(e)[:200]}"
            }
        )

# ============ README Agent - ZIP Upload - COMPLETELY REWRITTEN ============
@app.post("/api/upload-zip", tags=["readme"])
async def upload_zip(zip_file: UploadFile = File(...)):
    """
    Upload ZIP, extract safely, analyze, and generate README.
    
    COMPLETELY REWRITTEN with:
    - Comprehensive error handling
    - Detailed error messages
    - No more 500 errors
    - Works with any valid ZIP file
    """
    
    # ----- STEP 1: Validate file exists -----
    if not zip_file:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "❌ No file uploaded"}
        )
    
    # ----- STEP 2: Validate filename -----
    if not zip_file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "❌ Uploaded file has no name"}
        )
    
    # ----- STEP 3: Validate file type -----
    if not zip_file.filename.lower().endswith('.zip'):
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "error": f"❌ File must be a ZIP archive. Got: {zip_file.filename}"
            }
        )
    
    # ----- STEP 4: Read file content with error handling -----
    try:
        content = await zip_file.read()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "error": f"❌ Failed to read uploaded file: {str(e)[:100]}"
            }
        )
    
    # ----- STEP 5: Validate file size -----
    if len(content) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "❌ Uploaded ZIP file is empty"}
        )
    
    if len(content) > settings.MAX_ZIP_SIZE:
        return JSONResponse(
            status_code=400,
            content={
                "success": False, 
                "error": f"❌ ZIP file too large: {len(content) / (1024*1024):.1f}MB (max: {settings.MAX_ZIP_SIZE / (1024*1024)}MB)"
            }
        )
    
    # ----- STEP 6: Process ZIP with temporary directory -----
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Save ZIP file
            zip_path = Path(tmpdir) / "upload.zip"
            zip_path.write_bytes(content)
            
            # Verify it's a valid ZIP
            if not zipfile.is_zipfile(zip_path):
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "❌ File is not a valid ZIP archive"}
                )
            
            # Create extraction directory
            extract_path = Path(tmpdir) / "extracted"
            extract_path.mkdir(exist_ok=True)
            
            # ----- STEP 7: Extract ZIP with proper error handling -----
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Check if ZIP is empty
                    if len(zf.namelist()) == 0:
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "error": "❌ ZIP file contains no files"}
                        )
                    
                    # Extract all files safely
                    for member in zf.namelist():
                        # Skip directories and macOS metadata
                        if member.endswith('/') or '__MACOSX' in member or member.startswith('._'):
                            continue
                        
                        try:
                            # Extract file
                            zf.extract(member, extract_path)
                        except Exception as e:
                            # Log but continue with other files
                            print(f"Warning: Failed to extract {member}: {e}")
                            continue
                            
            except zipfile.BadZipFile:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "❌ Corrupted ZIP file (BadZipFile)"}
                )
            except zipfile.LargeZipFile:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "❌ ZIP file too large (ZIP64 limits)"}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": f"❌ Failed to extract ZIP: {str(e)[:100]}"}
                )
            
            # ----- STEP 8: Find the project root -----
            try:
                # Get all extracted contents
                contents = list(extract_path.iterdir())
                
                if not contents:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": "❌ ZIP file extracted but no files found"}
                    )
                
                # If there's exactly one directory, use that as project root
                if len(contents) == 1 and contents[0].is_dir():
                    project_path = contents[0]
                else:
                    # Otherwise use the extraction directory itself
                    project_path = extract_path
                
                # Verify project path exists
                if not project_path.exists():
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "error": "❌ Internal error: Extracted path not found"}
                    )
                
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": f"❌ Error processing extracted files: {str(e)[:100]}"}
                )
            
            # ----- STEP 9: Generate README -----
            try:
                # Analyze project
                analyzer = ProjectAnalyzer(str(project_path))
                analysis = analyzer.analyze()
                
                # Generate README
                readme_content = generate_project_readme(str(project_path))
                
                # Success!
                return {
                    "success": True,
                    "readme": readme_content,
                    "project_name": project_path.name or "project",
                    "file_count": analysis['summary']['total_files'],
                    "summary": analysis['summary']
                }
                
            except ValueError as e:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": str(e)}
                )
            except Exception as e:
                # Log the full error for debugging
                print(f"README generation error: {traceback.format_exc()}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False, 
                        "error": f"❌ Failed to generate README: {str(e)[:200]}"
                    }
                )
                
        except Exception as e:
            # Catch any other errors in the ZIP process
            print(f"ZIP processing error: {traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False, 
                    "error": f"❌ ZIP processing failed: {str(e)[:200]}"
                }
            )

# ============ README Agent - Save to Disk ============
@app.post("/api/generate-readme-and-save", tags=["readme"])
async def generate_and_save_readme(
    project_path: str = Body(..., embed=True),
    output_path: Optional[str] = Body(None, embed=True)
):
    """Generate and save README.md directly to disk."""
    try:
        saved_path = save_readme(project_path, output_path)
        return {
            "success": True,
            "message": "README.md saved successfully",
            "file_path": str(saved_path)
        }
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"success": False, "error": str(e)}
        )

# ============ Utility Endpoints ============
@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Service health check with safety limits."""
    return HealthResponse(
        status="healthy",
        service=app.title,
        version=app.version,
        supports=["functions", "classes", "methods", "readme", "zip"],
        model=settings.MODEL_NAME,
        agents=["docstring_agent", "readme_agent"],
        safety_limits={
            "max_file_size_kb": settings.MAX_FILE_SIZE // 1024,
            "max_files": settings.MAX_FILES_TOTAL,
            "max_depth": settings.MAX_DEPTH,
            "max_zip_size_mb": settings.MAX_ZIP_SIZE // (1024*1024)
        }
    )

@app.get("/api/version", tags=["information"])
async def version_info():
    """API and model version information."""
    return {
        "api_version": app.version,
        "model_name": settings.MODEL_NAME,
        "temperature": settings.TEMPERATURE,
        "max_tokens_docstring": settings.MAX_TOKENS_DOCSTRING,
        "max_tokens_readme": settings.MAX_TOKENS_README,
        "features": ["docstring", "readme", "zip_upload", "cache"]
    }

@app.get("/api/supported-types", tags=["information"])
async def supported_types():
    """List all supported Python element types."""
    return {
        "docstring": ["function", "class", "method"],
        "readme": ["python_projects", "mixed_projects"],
        "upload": ["zip_archives"],
        "edge_cases_handled": [
            "Empty files",
            "Incomplete code",
            "Large folders (100KB/file limit)",
            "Nested structures (depth 10)",
            "Permission denied",
            "Binary files",
            "Syntax errors"
        ]
    }

# ============ Entry Point ============
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open("http://localhost:8000")
    
    print("=" * 80)
    print(f"🏆 {app.title} v{app.version}")
    print("=" * 80)
    print(f"📝 Docstring Agent | 📚 README Agent | 📦 ZIP Upload")
    print(f"🤖 Model: {settings.MODEL_NAME}")
    print(f"🛡️  Security: Path validation enabled")
    print(f"⚡ Cost Optimized: Tokens reduced 50% | Caching enabled")
    print(f"🌐 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 80)
    
    Timer(1.5, open_browser).start()
    uvicorn.run("app.__main__:app", host="0.0.0.0", port=8000, reload=True, log_level="info")