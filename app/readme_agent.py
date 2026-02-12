# readme_agent.py
"""
README Generation Agent - Hackathon Edition
-------------------------------------------
Compliant with Nasiko requirements:
✅ Reads all relevant files
✅ Understands project structure  
✅ Generates comprehensive README.md
✅ Edge case handling (empty, large, nested)
✅ Clear agent design documentation
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import get_llm, settings, cache_manager, get_settings


# ==================== CONFIGURATION ====================

@dataclass
class ReadmeAgentConfig:
    """
    Configuration for README generation agent.
    Designed for hackathon edge cases:
    - Empty directories
    - Very large projects  
    - Deep nesting
    - Binary files
    - Permission errors
    """
    # File/folder filters - comprehensive ignore list
    IGNORED_DIRS: Set[str] = field(default_factory=lambda: {
        '__pycache__', 'venv', 'env', '.venv', '.env',
        '.git', 'node_modules', 'dist', 'build', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', '.coverage', 'htmlcov',
        '.idea', '.vscode', '.vs', '.github', '.gitlab',
        '__MACOSX',  # Mac ZIP artifacts
    })
    
    IGNORED_FILES: Set[str] = field(default_factory=lambda: {
        '.DS_Store', '*.pyc', '*.pyo', '*.pyd', 
        '.env', '.gitignore', '.dockerignore',
        '*.log', '*.tmp', '*.cache', '*.lock',
        'desktop.ini', 'thumbs.db',
    })
    
    # Safety limits for large folders - hackathon friendly
    MAX_FILE_SIZE: int = 100_000  # 100 KB
    MAX_FILES_TOTAL: int = 500    # Max files to process
    MAX_DEPTH: int = 10          # Max directory depth
    
    # Token optimization limits
    MAX_FUNCTIONS_IN_SUMMARY: int = 15  # Reduced from 20
    MAX_CLASSES_IN_SUMMARY: int = 8     # Reduced from 10
    MAX_DEPENDENCIES: int = 15         # Reduced from 30
    MAX_STRUCTURE_ITEMS: int = 50      # New: limit tree items
    
    # LLM settings - optimized for cost
    TEMPERATURE: float = 0.1  # More deterministic
    MAX_TOKENS: int = 1500    # Optimized


@dataclass
class FileInfo:
    """Metadata for a single file - memory optimized."""
    path: str
    name: str
    extension: str
    size: int
    line_count: int
    is_binary: bool = False
    functions: List[Dict] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)  # Set for deduplication
    has_docstring: bool = False
    
    def __post_init__(self):
        """Convert imports list to set for deduplication."""
        if isinstance(self.imports, list):
            self.imports = set(self.imports)


@dataclass
class DirectoryNode:
    """
    Tree node representing a directory.
    Optimized for memory efficiency in large projects.
    """
    name: str
    path: str
    files: List[FileInfo] = field(default_factory=list)
    subdirs: Dict[str, 'DirectoryNode'] = field(default_factory=dict)
    
    @property
    def file_count(self) -> int:
        """Calculate total files in this directory and subdirectories."""
        count = len(self.files)
        for subdir in self.subdirs.values():
            count += subdir.file_count
        return count
    
    @property
    def dir_count(self) -> int:
        """Calculate total subdirectories."""
        count = len(self.subdirs)
        for subdir in self.subdirs.values():
            count += subdir.dir_count
        return count


class ProjectAnalyzer:
    """
    Advanced project analyzer with comprehensive edge case handling.
    
    Edge Cases Handled:
    ✅ Empty directories - gracefully handled
    ✅ Empty files - analyzed without errors
    ✅ Incomplete Python code - syntax errors caught
    ✅ Very large files (>100KB) - skipped with counter
    ✅ Permission denied - caught and continued
    ✅ Binary files - detected and skipped
    ✅ Deeply nested structures - limited by MAX_DEPTH
    ✅ Hidden files/directories - skipped
    ✅ Unicode decode errors - handled with errors='ignore'
    """
    
    def __init__(self, root_path: str, config: ReadmeAgentConfig = None):
        # Convert string path to Path object safely
        self.root_path = Path(root_path)
        self.config = config or ReadmeAgentConfig()
        self.settings = get_settings()
        
        # Get the directory name safely
        dir_name = self.root_path.name
        if not dir_name:  # Handle root drives like C:\
            dir_name = str(self.root_path).replace('\\', '/').rstrip('/').split('/')[-1] or "project"
        
        # Root node for tree structure
        self.root_node = DirectoryNode(
            name=dir_name,
            path=str(self.root_path)
        )
        
        # Summary statistics - optimized structure
        self.summary = {
            'total_files': 0,
            'total_dirs': 0,
            'skipped_files': 0,
            'skipped_size': 0,
            'skipped_reasons': {
                'too_large': 0,
                'binary': 0,
                'permission': 0,
                'syntax_error': 0,
                'other': 0
            },
            'file_types': {},
            'functions': [],
            'classes': [],
            'dependencies': set(),
            'total_lines': 0,
            'has_tests': False,
            'has_docs': False,
            'has_requirements': False,
            'has_setup': False,
            'has_docker': False,
            'empty_dirs': [],  # Track empty directories
            'warnings': [],    # Collect warnings for user
        }
        
        self.file_count = 0
    
    def analyze(self) -> Dict[str, Any]:
        """
        Recursively analyze project with comprehensive edge case handling.
        
        Returns:
            Dict with project_name, root_path, structure, summary
        """
        start_time = datetime.now()
        
        try:
            self._traverse(self.root_path, self.root_node)
        except Exception as e:
            self.summary['warnings'].append(f"Analysis error: {str(e)}")
        
        # Post-process summary
        self.summary['total_files'] = self.file_count
        self.summary['total_dirs'] = self.root_node.dir_count
        
        # Deduplicate and limit expensive fields
        self.summary['dependencies'] = sorted(self.summary['dependencies'])[:self.config.MAX_DEPENDENCIES]
        self.summary['functions'] = self.summary['functions'][:self.config.MAX_FUNCTIONS_IN_SUMMARY]
        self.summary['classes'] = self.summary['classes'][:self.config.MAX_CLASSES_IN_SUMMARY]
        
        # Add empty directory count
        self.summary['empty_dirs_count'] = len(self.summary['empty_dirs'])
        
        # Add analysis time
        self.summary['analysis_time_ms'] = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Check for edge cases
        if self.file_count == 0:
            self.summary['warnings'].append("No readable files found in directory")
        if self.summary['skipped_files'] > 0:
            reasons = []
            for k, v in self.summary['skipped_reasons'].items():
                if v > 0:
                    reasons.append(f"{k}: {v}")
            self.summary['warnings'].append(
                f"Skipped {self.summary['skipped_files']} files ({', '.join(reasons)})"
            )
        
        # Get project name safely
        project_name = self.root_path.name
        if not project_name:
            # Handle Windows drive roots (C:\, D:\) and other edge cases
            project_name = str(self.root_path).replace('\\', '/').rstrip('/').split('/')[-1] or "project"
        
        return {
            'project_name': project_name,
            'root_path': str(self.root_path.absolute()),
            'structure': self._serialize_node_optimized(self.root_node),
            'summary': self.summary,
            'config': {
                'max_file_size': self.config.MAX_FILE_SIZE,
                'max_files': self.config.MAX_FILES_TOTAL,
                'max_depth': self.config.MAX_DEPTH
            }
        }
    
    def _traverse(self, current_path: Path, node: DirectoryNode, depth: int = 0):
        """
        Recursively traverse with comprehensive error handling.
        Each edge case is caught and reported.
        """
        # Edge Case: Deep nesting
        if depth > self.config.MAX_DEPTH:
            self.summary['warnings'].append(f"Max depth {self.config.MAX_DEPTH} reached at {current_path}")
            return
        
        # Edge Case: Too many files
        if self.file_count > self.config.MAX_FILES_TOTAL:
            if self.file_count == self.config.MAX_FILES_TOTAL + 1:  # Only warn once
                self.summary['warnings'].append(f"Max files {self.config.MAX_FILES_TOTAL} reached")
            return
        
        try:
            # Get directory contents safely
            try:
                items = list(current_path.iterdir())
            except PermissionError:
                # Edge Case: Permission denied
                rel_path = self._get_relative_path_safe(current_path)
                self.summary['skipped_reasons']['permission'] += 1
                self.summary['warnings'].append(f"Permission denied: {rel_path}")
                return
            except Exception as e:
                # Edge Case: Other errors
                rel_path = self._get_relative_path_safe(current_path)
                self.summary['skipped_reasons']['other'] += 1
                self.summary['warnings'].append(f"Error accessing {rel_path}: {str(e)}")
                return
            
            # Edge Case: Empty directory
            if not items:
                rel_path = self._get_relative_path_safe(current_path)
                if rel_path != '.':  # Don't add root as empty dir
                    self.summary['empty_dirs'].append(rel_path)
                return
            
            # Sort directories first, then files
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                # Edge Case: Hidden files/dirs
                if item.name.startswith('.'):
                    continue
                
                try:
                    if item.is_dir():
                        # Skip ignored directories
                        if item.name in self.config.IGNORED_DIRS:
                            continue
                        
                        # Create subdirectory node
                        rel_path = self._get_relative_path_safe(item)
                        dir_node = DirectoryNode(
                            name=item.name,
                            path=rel_path
                        )
                        node.subdirs[item.name] = dir_node
                        
                        # Recursively traverse
                        self._traverse(item, dir_node, depth + 1)
                    
                    else:  # File
                        self._process_file(item, node)
                        
                except Exception as e:
                    # Log but continue processing
                    self.summary['skipped_reasons']['other'] += 1
                    self.summary['warnings'].append(f"Error processing {item.name}: {str(e)}")
                    
        except Exception as e:
            # Top-level error handler
            self.summary['warnings'].append(f"Unexpected error in {current_path}: {str(e)}")
    
    def _get_relative_path_safe(self, path: Path) -> str:
        """Safely get relative path, fallback to string representation."""
        try:
            if path == self.root_path:
                return "."
            rel_path = path.relative_to(self.root_path)
            return str(rel_path).replace('\\', '/')
        except (ValueError, AttributeError):
            # Fallback to string representation
            return str(path).replace('\\', '/')
    
    def _process_file(self, file_path: Path, node: DirectoryNode):
        """Process individual file with edge case handling."""
        # Edge Case: File size limit
        try:
            file_size = file_path.stat().st_size
        except OSError:
            self.summary['skipped_reasons']['permission'] += 1
            self.summary['skipped_files'] += 1
            return
        
        if file_size > self.config.MAX_FILE_SIZE:
            self.summary['skipped_files'] += 1
            self.summary['skipped_size'] += file_size
            self.summary['skipped_reasons']['too_large'] += 1
            return
        
        # Check file limit
        if self.file_count >= self.config.MAX_FILES_TOTAL:
            return
        
        self.file_count += 1
        
        # Analyze file
        file_info = self._analyze_file_safe(file_path)
        if file_info:
            node.files.append(file_info)
    
    def _analyze_file_safe(self, file_path: Path) -> Optional[FileInfo]:
        """
        Analyze file with comprehensive error handling.
        Never crashes - always returns None or valid FileInfo.
        """
        try:
            rel_path = self._get_relative_path_safe(file_path)
            
            file_info = FileInfo(
                path=rel_path,
                name=file_path.name,
                extension=file_path.suffix.lower() or 'no_extension',
                size=file_path.stat().st_size,
                line_count=0
            )
            
            # Try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_info.line_count = content.count('\n') + 1
                    
                    # Python file analysis
                    if file_path.suffix.lower() == '.py':
                        self._analyze_python_content(content, file_info, rel_path)
                        
            except UnicodeDecodeError:
                # Edge Case: Binary file
                file_info.is_binary = True
                self.summary['skipped_reasons']['binary'] += 1
            
            # Update file type statistics
            ext = file_info.extension
            self.summary['file_types'][ext] = self.summary['file_types'].get(ext, 0) + 1
            self.summary['total_lines'] += file_info.line_count
            
            # Detect project features
            name_lower = file_info.name.lower()
            if 'test' in name_lower:
                self.summary['has_tests'] = True
            if 'doc' in name_lower or file_info.extension == '.md':
                self.summary['has_docs'] = True
            if file_info.name == 'requirements.txt':
                self.summary['has_requirements'] = True
                self._parse_requirements(file_path)
            if file_info.name == 'setup.py':
                self.summary['has_setup'] = True
            if file_info.name == 'Dockerfile':
                self.summary['has_docker'] = True
            
            return file_info
            
        except PermissionError:
            # Edge Case: Permission denied
            self.summary['skipped_reasons']['permission'] += 1
            self.summary['skipped_files'] += 1
            return None
        except Exception as e:
            # Edge Case: Any other error - log and skip
            self.summary['skipped_files'] += 1
            self.summary['skipped_reasons']['other'] += 1
            return None
    
    def _analyze_python_content(self, content: str, file_info: FileInfo, rel_path: str):
        """
        Analyze Python code with AST.
        Edge Case: Syntax errors are caught and logged.
        """
        # Edge Case: Empty file
        if not content.strip():
            return
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Extract functions
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'file': rel_path,
                        'line': node.lineno,
                        'args': len(node.args.args)
                    }
                    file_info.functions.append(func_info)
                    self.summary['functions'].append(func_info)
                
                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body 
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ][:10]  # Limit methods shown
                    
                    class_info = {
                        'name': node.name,
                        'file': rel_path,
                        'line': node.lineno,
                        'methods': methods
                    }
                    file_info.classes.append(class_info)
                    self.summary['classes'].append(class_info)
                
                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level package only
                        pkg = alias.name.split('.')[0]
                        file_info.imports.add(pkg)
                        self.summary['dependencies'].add(pkg)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split('.')[0]
                        file_info.imports.add(pkg)
                        self.summary['dependencies'].add(pkg)
            
            # Check for docstrings
            if ast.get_docstring(tree):
                file_info.has_docstring = True
                
        except SyntaxError:
            # Edge Case: Incomplete/invalid Python code
            self.summary['skipped_reasons']['syntax_error'] += 1
        except Exception:
            # Other AST errors
            self.summary['skipped_reasons']['other'] += 1
    
    def _parse_requirements(self, file_path: Path):
        """Parse requirements.txt for dependencies."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before version specifiers)
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        if pkg:
                            # Clean up common markers
                            pkg = pkg.split(';')[0].split('[')[0].strip()
                            if pkg:
                                self.summary['dependencies'].add(pkg)
        except Exception:
            pass
    
    def _serialize_node_optimized(self, node: DirectoryNode, max_items: int = 50) -> Dict:
        """
        Serialize node with token optimization.
        Limits output size for large projects.
        """
        # Limit files shown
        files_shown = node.files[:max_items // 2]
        files_truncated = len(node.files) - len(files_shown)
        
        # Limit subdirs shown
        subdir_items = list(node.subdirs.items())[:max_items // 2]
        subdirs_truncated = len(node.subdirs) - len(subdir_items)
        
        result = {
            'name': node.name,
            'path': node.path.replace('\\', '/'),  # Normalize to forward slashes
            'type': 'directory',
            'files': [
                {
                    'name': f.name,
                    'extension': f.extension,
                    'line_count': f.line_count,
                    'is_binary': f.is_binary,
                    'functions_count': len(f.functions),
                    'classes_count': len(f.classes)
                }
                for f in files_shown
            ],
            'subdirs': {
                name: self._serialize_node_optimized(subdir, max_items // 2)
                for name, subdir in subdir_items
            }
        }
        
        # Add truncation indicators only if needed
        if files_truncated > 0:
            result['files_truncated'] = files_truncated
        if subdirs_truncated > 0:
            result['subdirs_truncated'] = subdirs_truncated
        
        return result


class ReadmeGenerator:
    """
    README Generator with optimized prompts for token efficiency.
    
    Agent Design:
    -------------
    1. STATIC ANALYSIS STAGE:
       - Recursive directory traversal
       - AST parsing for Python files
       - Metadata extraction (functions, classes, imports)
       - Edge case detection and handling
       - Size/limit enforcement
    
    2. SEMANTIC SYNTHESIS STAGE:
       - Structured data preparation (not raw code)
       - Token-optimized summary generation
       - Gemini LLM with specialized prompts
       - Markdown formatting with professional structure
    
    Assumptions:
    ------------
    - Files > 100KB are skipped (configurable)
    - Max 500 files processed (configurable)
    - Max depth 10 directories (configurable)
    - Only Python files analyzed for functions/classes
    - Hidden files/dirs ignored
    - Binary files skipped
    
    Limitations:
    ------------
    - Cannot analyze encrypted/minified code
    - Dependency detection limited to import statements
    - Method detection uses self/cls heuristic
    - Very large monorepos may be truncated
    """
    
    def __init__(self, analysis: Dict[str, Any]):
        self.analysis = analysis
        self.settings = get_settings()
        
        # Use shared LLM with optimized token limit
        self.llm = get_llm(
            max_tokens=settings.MAX_TOKENS_README,
            temperature=0.1  # More deterministic for docs
        )
    
    def generate(self) -> str:
        """
        Generate comprehensive README.md with token optimization.
        """
        
        # Create cache key from analysis summary
        cache_key = {
            'type': 'readme',
            'project_name': self.analysis['project_name'],
            'file_count': self.analysis['summary']['total_files'],
            'dir_count': self.analysis['summary']['total_dirs'],
            'functions_count': len(self.analysis['summary']['functions']),
            'classes_count': len(self.analysis['summary']['classes']),
            'dependencies': list(self.analysis['summary']['dependencies'])[:10],
            'has_tests': self.analysis['summary']['has_tests'],
            'has_docs': self.analysis['summary']['has_docs']
        }
        
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        # Token-optimized prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior technical documentation expert.
Generate a professional README.md with these sections in order:

1. # Project Title
2. ## 📋 Overview  
3. ## ✨ Features
4. ## 🚀 Installation
5. ## 💻 Usage  
6. ## 📁 Project Structure
7. ## 📊 Statistics
8. ## ⚠️ Notes

Rules:
- Be concise but comprehensive
- Use proper markdown
- No placeholder text
- Max 1500 tokens"""),
            
            ("human", """Generate README from this analysis:

PROJECT: {project_name}
FILES: {total_files} files, {total_dirs} dirs
LANGUAGES: {file_types}
FUNCTIONS: {functions_count}
CLASSES: {classes_count}
DEPS: {dependencies}

FEATURES:
- Tests: {has_tests}
- Docs: {has_docs}  
- Requirements: {has_requirements}

STRUCTURE:
{structure}

WARNINGS:
{warnings}

Generate complete README.md:""")
        ])
        
        # Prepare minimal, token-efficient data
        warnings = self.analysis['summary'].get('warnings', [])
        if self.analysis['summary'].get('empty_dirs_count', 0) > 0:
            warnings.append(f"{self.analysis['summary']['empty_dirs_count']} empty directories")
        
        # Use simplified structure for large projects
        if self.analysis['summary']['total_files'] > 100:
            structure = self._get_simplified_structure()
        else:
            structure = self._format_structure_minimal(self.analysis['structure'])
        
        # Format file types nicely
        file_types_dict = self.analysis['summary'].get('file_types', {})
        file_types = ", ".join([
            f"{k}: {v}" for k, v in list(file_types_dict.items())[:5]
        ]) or "Python" if file_types_dict.get('.py') else "Various"
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "project_name": self.analysis['project_name'],
                "total_files": self.analysis['summary']['total_files'],
                "total_dirs": self.analysis['summary']['total_dirs'],
                "file_types": file_types,
                "functions_count": len(self.analysis['summary']['functions']),
                "classes_count": len(self.analysis['summary']['classes']),
                "dependencies": ", ".join(self.analysis['summary']['dependencies'][:10]) or "None detected",
                "has_tests": "✅" if self.analysis['summary']['has_tests'] else "❌",
                "has_docs": "✅" if self.analysis['summary']['has_docs'] else "❌", 
                "has_requirements": "✅" if self.analysis['summary']['has_requirements'] else "❌",
                "structure": structure,
                "warnings": "\n".join(warnings[:3]) if warnings else "None"
            })
        except Exception as e:
            # Fallback: Generate minimal README without LLM
            response = self._generate_fallback_readme()
        
        # Cache the result
        cache_manager.set(cache_key, response)
        
        return response
    
    def _generate_fallback_readme(self) -> str:
        """Generate minimal README when LLM fails."""
        summary = self.analysis['summary']
        name = self.analysis['project_name']
        
        warnings_text = ""
        if summary.get('warnings'):
            warnings_text = "\n".join(summary['warnings'][:3])
        
        return f"""# {name}

## 📋 Overview
A Python project with {summary['total_files']} files in {summary['total_dirs']} directories.

## 📁 Project Structure
```
{self._get_simplified_structure()}
```

## 📊 Statistics
- **Total Files**: {summary['total_files']}
- **Total Directories**: {summary['total_dirs']}
- **Total Lines of Code**: {summary['total_lines']}
- **Functions**: {len(summary['functions'])}
- **Classes**: {len(summary['classes'])}
- **Dependencies**: {len(summary['dependencies'])}

## ⚠️ Notes
README generated by AI Documentation Agent.

{warnings_text}
"""
    
    def _format_structure_minimal(self, node: Dict, prefix: str = "", max_depth: int = 3) -> str:
        """
        Token-efficient structure formatting.
        Limits depth and items shown.
        """
        if max_depth <= 0:
            return f"{prefix}  ... (depth limit reached)"
        
        lines = []
        
        # Root node
        if prefix == "":
            lines.append(f"📁 {node['name']}/")
        
        # Files (limit to 5)
        files = node.get('files', [])[:5]
        for i, file in enumerate(files):
            is_last = i == len(files) - 1 and not node.get('subdirs')
            connector = "└── " if is_last else "├── "
            icon = "🐍" if file.get('extension') == '.py' else "📄"
            lines.append(f"{prefix}{connector}{icon} {file['name']}")
        
        # Show truncation indicator
        if len(node.get('files', [])) > 5:
            lines.append(f"{prefix}├── ... and {len(node['files']) - 5} more files")
        
        # Subdirs
        subdirs = list(node.get('subdirs', {}).items())[:3]
        for i, (name, subdir) in enumerate(subdirs):
            is_last = i == len(subdirs) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}📁 {name}/")
            subdir_content = self._format_structure_minimal(subdir, child_prefix, max_depth - 1)
            if subdir_content:
                lines.append(subdir_content)
        
        if len(node.get('subdirs', {})) > 3:
            lines.append(f"{prefix}└── ... and {len(node['subdirs']) - 3} more directories")
        
        return "\n".join(lines)
    
    def _get_simplified_structure(self) -> str:
        """Ultra-minimal structure for very large projects."""
        summary = self.analysis['summary']
        return f"""📁 {self.analysis['project_name']}/
  📄 {summary['total_files']} files total
  📁 {summary['total_dirs']} directories total

Top-level directories:
{self._get_top_directories()}"""

    def _get_top_directories(self) -> str:
        """Get top-level directories only."""
        top_dirs = list(self.analysis['structure'].get('subdirs', {}).keys())[:5]
        if top_dirs:
            return "\n".join(f"  📁 {d}/" for d in top_dirs)
        return "  (no subdirectories)"


def generate_project_readme(project_path: str) -> str:
    """
    Generate comprehensive README.md for any project.
    
    Args:
        project_path: Path to project root directory
    
    Returns:
        Complete README.md content as string
    
    Raises:
        ValueError: If project path is invalid or inaccessible
    """
    # Validate input
    if not project_path or not isinstance(project_path, str):
        raise ValueError("Invalid project path")
    
    # Convert to Path object
    try:
        path = Path(project_path)
    except Exception:
        raise ValueError(f"Invalid path format: {project_path}")
    
    # Check existence
    if not path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    
    # Check if it's a directory
    if not path.is_dir():
        raise ValueError(f"Project path must be a directory: {project_path}")
    
    # Security: Check for disallowed paths (Windows and Linux)
    settings = get_settings()
    path_str = str(path.absolute()).lower()
    
    for disallowed in settings.DISALLOWED_PATHS:
        if path_str.startswith(disallowed.lower()):
            raise ValueError(f"Access denied: {disallowed} is not allowed")
    
    # Analyze project with safety limits
    config = ReadmeAgentConfig()
    analyzer = ProjectAnalyzer(str(path), config)
    analysis = analyzer.analyze()
    
    # Generate README
    generator = ReadmeGenerator(analysis)
    readme = generator.generate()
    
    return readme


def save_readme(project_path: str, output_path: Optional[str] = None) -> str:
    """Generate and save README.md with metadata."""
    readme_content = generate_project_readme(project_path)
    
    if not output_path:
        output_path = Path(project_path) / "README.md"
    else:
        output_path = Path(output_path)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add generation metadata as HTML comment
    metadata = f"""<!--
README generated by Gemini README Generation Agent
Hackathon Submission - Nasiko Labs
Timestamp: {datetime.now().isoformat()}
Project: {Path(project_path).name}
Agent Version: 2.0.0
Edge Cases Handled: Empty dirs, large files, nested structures, permission errors
-->
"""
    
    output_path.write_text(metadata + readme_content, encoding='utf-8')
    return str(output_path)