"""
README Generation Agent - Analyzes and explains user's project architecture
Uses Gemini to understand what the project does and generate meaningful diagrams
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import get_llm, settings


# ==================== CONFIG ====================

@dataclass
class ReadmeAgentConfig:
    """Configuration for README generation agent."""
    
    IGNORED_DIRS: Set[str] = field(default_factory=lambda: {
        '__pycache__', 'venv', 'env', '.venv', '.env', '.git', 'node_modules',
        'dist', 'build', '.pytest_cache', '.mypy_cache', '__MACOSX'
    })
    
    IGNORED_FILES: Set[str] = field(default_factory=lambda: {
        '.DS_Store', '*.pyc', '*.pyo', '.env', '.gitignore',
        '*.log', '*.lock', 'desktop.ini', 'thumbs.db'
    })
    
    MAX_FILE_SIZE: int = 200_000
    MAX_FILES_TOTAL: int = 500
    MAX_DEPTH: int = 10


@dataclass
class FunctionInfo:
    name: str
    file: str
    line: int
    args: List[str]
    has_docstring: bool
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    file: str
    line: int
    bases: List[str]
    methods: List[str] = field(default_factory=list)
    has_docstring: bool = False


@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size: int
    line_count: int
    is_binary: bool = False
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    is_test_file: bool = False
    is_init_file: bool = False


@dataclass
class DirectoryNode:
    name: str
    path: str
    files: List[FileInfo] = field(default_factory=list)
    subdirs: Dict[str, 'DirectoryNode'] = field(default_factory=dict)
    
    @property
    def file_count(self) -> int:
        count = len(self.files)
        for subdir in self.subdirs.values():
            count += subdir.file_count
        return count
    
    @property
    def dir_count(self) -> int:
        count = len(self.subdirs)
        for subdir in self.subdirs.values():
            count += subdir.dir_count
        return count


class ProjectAnalyzer:
    """
    Analyzes user's project and extracts comprehensive metadata
    for intelligent architecture understanding.
    """
    
    def __init__(self, root_path: str, config: ReadmeAgentConfig = None):
        self.root_path = Path(root_path).resolve()
        self.config = config or ReadmeAgentConfig()
        
        dir_name = self.root_path.name or "project"
        
        self.root_node = DirectoryNode(name=dir_name, path=".")
        
        # Project metadata
        self.summary = {
            'project_name': dir_name,
            'total_files': 0,
            'total_dirs': 0,
            'file_types': {},
            'classes': [],
            'functions': [],
            'class_hierarchy': defaultdict(list),
            'dependencies': set(),
            'total_lines': 0,
            'test_files': 0,
            'has_tests': False,
            'has_requirements': False,
            'entry_points': [],
            'skipped_files': 0,
            'frameworks': set(),
            'project_type': None,  # Will be determined by LLM
            'key_components': [],  # Will be determined by LLM
            'data_flow': [],      # Will be determined by LLM
        }
        
        self.file_count = 0
    
    def analyze(self) -> Dict[str, Any]:
        """Extract all metadata from the project."""
        try:
            self._traverse(self.root_path, self.root_node)
            self._detect_entry_points()
            self._detect_frameworks()
        except Exception as e:
            pass
        
        self.summary['total_files'] = self.file_count
        self.summary['total_dirs'] = self.root_node.dir_count
        self.summary['dependencies'] = sorted(self.summary['dependencies'])[:50]
        self.summary['class_hierarchy'] = dict(self.summary['class_hierarchy'])
        
        # Prepare data for LLM understanding
        project_data = {
            'project_name': self.summary['project_name'],
            'structure': self._serialize_node(self.root_node),
            'summary': self.summary,
            'classes': self.summary['classes'],
            'functions': self.summary['functions'],
            'dependencies': list(self.summary['dependencies']),
            'entry_points': self.summary['entry_points'],
            'file_tree': self._get_file_tree_summary()
        }
        
        return project_data
    
    def _traverse(self, current_path: Path, node: DirectoryNode, depth: int = 0):
        """Traverse directory structure."""
        if depth > self.config.MAX_DEPTH or self.file_count > self.config.MAX_FILES_TOTAL:
            return
        
        try:
            if not current_path.exists():
                return
            
            items = list(current_path.iterdir())
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items:
                if item.name.startswith('.') or item.name in self.config.IGNORED_DIRS:
                    continue
                
                try:
                    if item.is_dir():
                        rel_path = self._safe_relative_path(item)
                        dir_node = DirectoryNode(name=item.name, path=rel_path)
                        node.subdirs[item.name] = dir_node
                        self._traverse(item, dir_node, depth + 1)
                    else:
                        self._process_file(item, node)
                except Exception:
                    continue
        except Exception:
            pass
    
    def _safe_relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_path)).replace('\\', '/')
        except Exception:
            return path.name
    
    def _process_file(self, file_path: Path, node: DirectoryNode):
        """Process individual file."""
        try:
            file_size = file_path.stat().st_size
            if file_size > self.config.MAX_FILE_SIZE:
                self.summary['skipped_files'] += 1
                return
            
            if self.file_count >= self.config.MAX_FILES_TOTAL:
                return
            
            self.file_count += 1
            file_info = self._analyze_file(file_path)
            if file_info:
                node.files.append(file_info)
        except Exception:
            self.summary['skipped_files'] += 1
    
    def _analyze_file(self, file_path: Path) -> Optional[FileInfo]:
        """Analyze file content."""
        try:
            rel_path = self._safe_relative_path(file_path)
            
            file_info = FileInfo(
                path=rel_path,
                name=file_path.name,
                extension=file_path.suffix.lower() or '.txt',
                size=file_path.stat().st_size,
                line_count=0,
                is_test_file='test' in file_path.name.lower(),
                is_init_file=file_path.name == '__init__.py'
            )
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_info.line_count = content.count('\n') + 1
                    
                    if file_path.suffix.lower() == '.py':
                        self._parse_python(content, file_info, rel_path)
            except UnicodeDecodeError:
                file_info.is_binary = True
            
            ext = file_info.extension
            self.summary['file_types'][ext] = self.summary['file_types'].get(ext, 0) + 1
            self.summary['total_lines'] += file_info.line_count
            
            if file_info.is_test_file:
                self.summary['test_files'] += 1
                self.summary['has_tests'] = True
            
            if file_info.name == 'requirements.txt' and 'content' in locals():
                self._parse_requirements(content)
            
            return file_info
        except Exception:
            return None
    
    def _parse_python(self, content: str, file_info: FileInfo, rel_path: str):
        """Parse Python file with AST."""
        if not content.strip():
            return
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                try:
                    if isinstance(node, ast.ClassDef):
                        self._parse_class(node, file_info, rel_path)
                    elif isinstance(node, ast.FunctionDef):
                        self._parse_function(node, file_info, rel_path, is_method=False)
                    elif isinstance(node, ast.AsyncFunctionDef):
                        self._parse_function(node, file_info, rel_path, is_method=False)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            pkg = alias.name.split('.')[0]
                            file_info.imports.add(pkg)
                            self.summary['dependencies'].add(pkg)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            pkg = node.module.split('.')[0]
                            file_info.imports.add(pkg)
                            self.summary['dependencies'].add(pkg)
                except Exception:
                    continue
        except SyntaxError:
            pass
    
    def _parse_class(self, node: ast.ClassDef, file_info: FileInfo, rel_path: str):
        """Parse class definition."""
        try:
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            
            methods = []
            decorators = []
            
            # Check for decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
            
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
                    self._parse_function(item, file_info, rel_path, 
                                       is_method=True, class_name=node.name)
            
            docstring = ast.get_docstring(node)
            
            class_info = ClassInfo(
                name=node.name,
                file=rel_path,
                line=getattr(node, 'lineno', 0),
                bases=bases,
                methods=methods[:20],
                has_docstring=docstring is not None
            )
            
            file_info.classes.append(class_info)
            
            self.summary['classes'].append({
                'name': node.name,
                'file': rel_path,
                'bases': bases,
                'methods_count': len(methods),
                'has_docstring': docstring is not None,
                'decorators': decorators
            })
            
            for base in bases:
                self.summary['class_hierarchy'][base].append(node.name)
        except Exception:
            pass
    
    def _parse_function(self, node: ast.FunctionDef, file_info: FileInfo, 
                       rel_path: str, is_method: bool = False, 
                       class_name: Optional[str] = None):
        """Parse function definition."""
        try:
            args = []
            for arg in node.args.args:
                if is_method and arg.arg in ['self', 'cls']:
                    continue
                args.append(arg.arg)
            
            # Get decorators
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
            
            docstring = ast.get_docstring(node)
            
            func_info = FunctionInfo(
                name=node.name,
                file=rel_path,
                line=getattr(node, 'lineno', 0),
                args=args[:15],
                has_docstring=docstring is not None,
                is_method=is_method,
                class_name=class_name,
                decorators=decorators
            )
            
            file_info.functions.append(func_info)
            
            self.summary['functions'].append({
                'name': node.name,
                'file': rel_path,
                'class': class_name,
                'args': args[:10],
                'has_docstring': docstring is not None,
                'is_method': is_method,
                'decorators': decorators,
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            })
        except Exception:
            pass
    
    def _parse_requirements(self, content: str):
        """Parse requirements.txt."""
        try:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                    pkg = pkg.split('[')[0].strip()
                    if pkg:
                        self.summary['dependencies'].add(pkg)
        except Exception:
            pass
    
    def _detect_entry_points(self):
        """Detect main entry points."""
        try:
            all_files = self._get_all_files()
            for file_info in all_files:
                if file_info.name in ['__main__.py', 'main.py', 'app.py', 'run.py', 'cli.py']:
                    self.summary['entry_points'].append({
                        'file': file_info.path,
                        'name': file_info.name
                    })
        except Exception:
            pass
    
    def _detect_frameworks(self):
        """Detect frameworks from dependencies and code patterns."""
        deps = self.summary['dependencies']
        
        if 'fastapi' in deps or 'uvicorn' in deps:
            self.summary['frameworks'].add('FastAPI')
        if 'flask' in deps:
            self.summary['frameworks'].add('Flask')
        if 'django' in deps:
            self.summary['frameworks'].add('Django')
        if 'sqlalchemy' in deps or 'django.db' in deps:
            self.summary['frameworks'].add('SQLAlchemy/Database')
        if 'graphene' in deps or 'strawberry' in deps:
            self.summary['frameworks'].add('GraphQL')
        if 'pytest' in deps or 'unittest' in deps:
            self.summary['has_tests'] = True
    
    def _get_all_files(self) -> List[FileInfo]:
        """Get all files."""
        files = []
        def collect(node: DirectoryNode):
            files.extend(node.files)
            for subdir in node.subdirs.values():
                collect(subdir)
        collect(self.root_node)
        return files
    
    def _serialize_node(self, node: DirectoryNode, max_items: int = 50) -> Dict:
        """Serialize node for structure tree."""
        result = {
            'name': node.name,
            'path': node.path,
            'files': [],
            'subdirs': {}
        }
        
        for f in node.files[:max_items]:
            result['files'].append({
                'name': f.name,
                'extension': f.extension,
                'is_test': f.is_test_file,
                'is_init': f.is_init_file,
                'classes_count': len(f.classes),
                'functions_count': len(f.functions)
            })
        
        for name, subdir in list(node.subdirs.items())[:max_items//2]:
            result['subdirs'][name] = self._serialize_node(subdir, max_items//2)
        
        return result
    
    def _get_file_tree_summary(self) -> str:
        """Get a text summary of the file tree for LLM."""
        lines = []
        
        def build_tree(node: DirectoryNode, prefix: str = ""):
            lines.append(f"{prefix}📁 {node.name}/")
            for f in node.files[:10]:
                lines.append(f"{prefix}  📄 {f.name}")
            for name, subdir in list(node.subdirs.items())[:5]:
                build_tree(subdir, prefix + "  ")
        
        build_tree(self.root_node)
        return "\n".join(lines[:50])


class ReadmeGenerator:
    """
    Uses Gemini to understand the project and generate intelligent documentation
    with architecture diagrams and meaningful explanations.
    """
    
    def __init__(self, project_data: Dict[str, Any]):
        self.project_data = project_data
        self.llm = get_llm(max_tokens=3000, temperature=0.2)
    
    def generate(self) -> str:
        """Generate intelligent README with architecture understanding."""
        
        # Prepare project summary for LLM
        summary = self.project_data['summary']
        classes = summary.get('classes', [])
        functions = summary.get('functions', [])
        deps = list(summary.get('dependencies', []))[:20]
        
        # Group classes by file for context
        classes_by_file = defaultdict(list)
        for cls in classes:
            classes_by_file[cls['file']].append(cls)
        
        # Group functions by file
        funcs_by_file = defaultdict(list)
        for func in functions:
            funcs_by_file[func['file']].append(func)
        
        # Create a prompt that asks Gemini to UNDERSTAND the project
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert software architect and technical writer.
Your task is to analyze a codebase and generate a comprehensive README.md that explains:

1. **What the project does** - The main purpose and functionality
2. **Architecture** - How components interact (with Mermaid diagrams)
3. **Data Flow** - How information moves through the system
4. **Key Components** - The important classes and what they do
5. **API/Interface** - How to use the project

IMPORTANT GUIDELINES:
- Be SPECIFIC to this project - use actual class/function names
- Create MEANINGFUL Mermaid diagrams that show real relationships
- Don't be generic - every section should reference actual code
- Focus on helping someone understand THIS specific codebase
"""),
            
            ("human", """Analyze this Python project and generate a comprehensive README.md:

PROJECT NAME: {project_name}
TOTAL FILES: {total_files}
TOTAL CLASSES: {total_classes}
TOTAL FUNCTIONS: {total_functions}
DETECTED FRAMEWORKS: {frameworks}

KEY DEPENDENCIES:
{deps}

CLASS HIERARCHIES (by file):
{class_hierarchies}

IMPORTANT FUNCTIONS (by file):
{key_functions}

ENTRY POINTS:
{entry_points}

PROJECT STRUCTURE:
{structure}

Based on this data, generate a README.md with:

1. **Project Title** - Use the actual project name

2. **📋 Overview** - 2-3 paragraphs explaining WHAT this project does and its main purpose. Be specific - reference actual classes and features.

3. **✨ Key Features** - Bullet points of actual capabilities based on the code

4. **🏗️ Architecture** - A Mermaid diagram showing the main components and how they interact. Use actual module/class names.

5. **📊 Data Flow** - A sequence diagram showing how a typical request flows through the system

6. **📦 Core Components** - Detailed explanation of the main classes and what they do

7. **🔌 API Reference** - Key functions/methods with their parameters

8. **🚀 Getting Started** - How to run/use this project

9. **📁 Project Structure** - Visual tree of important directories

Generate the README.md now:""")
        ])
        
        # Prepare the data
        class_hierarchies_text = ""
        for file_path, cls_list in list(classes_by_file.items())[:5]:
            class_hierarchies_text += f"\n{file_path}:\n"
            for cls in cls_list[:5]:
                bases = ", ".join(cls['bases']) if cls['bases'] else "object"
                class_hierarchies_text += f"  - {cls['name']} ({bases}) - {cls['methods_count']} methods\n"
        
        key_functions_text = ""
        for file_path, func_list in list(funcs_by_file.items())[:5]:
            key_functions_text += f"\n{file_path}:\n"
            for func in func_list[:8]:
                if func.get('is_method'):
                    key_functions_text += f"  - method: {func['class']}.{func['name']}({', '.join(func['args'])})\n"
                else:
                    key_functions_text += f"  - function: {func['name']}({', '.join(func['args'])})\n"
        
        entry_points_text = "\n".join([f"- {ep['file']}" for ep in summary.get('entry_points', [])]) or "None detected"
        
        # Generate structure tree
        structure_text = self._format_structure_preview(self.project_data['structure'])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            readme = chain.invoke({
                'project_name': summary['project_name'],
                'total_files': summary['total_files'],
                'total_classes': len(classes),
                'total_functions': len(functions),
                'frameworks': ', '.join(summary.get('frameworks', [])) or 'Python',
                'deps': '\n'.join([f"- {d}" for d in deps[:15]]),
                'class_hierarchies': class_hierarchies_text,
                'key_functions': key_functions_text,
                'entry_points': entry_points_text,
                'structure': structure_text
            })
            
            # Add the project structure tree at the end (always include this)
            readme += f"\n\n## 📁 Full Project Structure\n\n```\n{self._format_full_structure(self.project_data['structure'])}\n```\n"
            
            return readme
            
        except Exception as e:
            # Fallback to structural README if LLM fails
            return self._generate_structural_readme()
    
    def _format_structure_preview(self, node: Dict, depth: int = 0, max_depth: int = 2) -> str:
        """Format a preview of the structure (first 2 levels)."""
        if depth > max_depth:
            return ""
        
        lines = []
        indent = "  " * depth
        
        if depth == 0:
            lines.append(f"{indent}📁 {node['name']}/")
        
        # Show some files
        files = node.get('files', [])[:5]
        for f in files:
            lines.append(f"{indent}  📄 {f['name']}")
        
        # Show subdirectories
        for name, subdir in list(node.get('subdirs', {}).items())[:3]:
            lines.append(f"{indent}  📁 {name}/")
            if depth < max_depth:
                sub_preview = self._format_structure_preview(subdir, depth + 1, max_depth)
                if sub_preview:
                    lines.append(sub_preview)
        
        return "\n".join(lines)
    
    def _format_full_structure(self, node: Dict, prefix: str = "", is_last: bool = True) -> str:
        """Format full directory structure as a tree."""
        lines = []
        
        if prefix == "":
            lines.append(f"📁 {node['name']}/")
        
        files = node.get('files', [])
        for i, file in enumerate(files):
            is_last_file = (i == len(files) - 1) and not node.get('subdirs')
            connector = "└── " if is_last_file else "├── "
            icon = "🐍" if file.get('extension') == '.py' else "📄"
            test = " 🧪" if file.get('is_test') else ""
            init = " 📦" if file.get('is_init') else ""
            classes = f" [{file.get('classes_count', 0)}cls]" if file.get('classes_count', 0) > 0 else ""
            funcs = f" [{file.get('functions_count', 0)}fn]" if file.get('functions_count', 0) > 0 else ""
            lines.append(f"{prefix}{connector}{icon} {file['name']}{test}{init}{classes}{funcs}")
        
        subdirs = node.get('subdirs', {})
        subdir_items = list(subdirs.items())
        for i, (name, subdir) in enumerate(subdir_items):
            is_last_subdir = (i == len(subdir_items) - 1)
            connector = "└── " if is_last_subdir else "├── "
            lines.append(f"{prefix}{connector}📁 {name}/")
            child_prefix = prefix + ("    " if is_last_subdir else "│   ")
            lines.append(self._format_full_structure(subdir, child_prefix, is_last_subdir))
        
        return "\n".join(lines)
    
    def _generate_structural_readme(self) -> str:
        """Fallback README with structural information."""
        summary = self.project_data['summary']
        name = summary['project_name']
        
        sections = []
        sections.append(f"# {name}\n")
        
        sections.append("## 📋 Overview\n")
        sections.append(f"A Python project with **{summary['total_files']} files** in **{summary['total_dirs']} directories**.")
        if summary['total_lines'] > 0:
            sections.append(f"\nTotal lines of code: **{summary['total_lines']:,}**")
        sections.append("")
        
        # Features
        features = []
        if summary['classes']:
            features.append(f"- 🏛️ **{len(summary['classes'])} Classes**")
        if summary['functions']:
            func_count = len([f for f in summary['functions'] if not f.get('is_method')])
            method_count = len([f for f in summary['functions'] if f.get('is_method')])
            features.append(f"- ⚡ **{func_count} Functions** and **{method_count} Methods**")
        if summary['has_tests']:
            features.append(f"- 🧪 **{summary['test_files']} Test Files**")
        if summary['entry_points']:
            features.append(f"- 🚀 **{len(summary['entry_points'])} Entry Points**")
        
        if features:
            sections.append("## ✨ Features\n")
            sections.append("\n".join(features))
            sections.append("")
        
        # Dependencies
        if summary['dependencies']:
            sections.append("## 📦 Dependencies\n")
            for dep in sorted(summary['dependencies'])[:30]:
                sections.append(f"- `{dep}`")
            sections.append("")
        
        # Project Structure
        sections.append("## 📁 Project Structure\n")
        sections.append("```")
        sections.append(self._format_full_structure(self.project_data['structure']))
        sections.append("```\n")
        
        return "\n".join(sections)


def generate_project_readme(project_path: str) -> str:
    """
    Generate an intelligent README.md that actually understands the project's architecture.
    Uses Gemini to analyze what the project does and generate meaningful diagrams.
    """
    try:
        path = Path(project_path).resolve()
        if not path.exists() or not path.is_dir():
            return f"# {path.name}\n\nUnable to analyze project: Path not accessible."
        
        # Analyze the project
        config = ReadmeAgentConfig()
        analyzer = ProjectAnalyzer(str(path), config)
        project_data = analyzer.analyze()
        
        # Generate intelligent README using Gemini
        generator = ReadmeGenerator(project_data)
        readme = generator.generate()
        
        return readme
        
    except Exception as e:
        return f"# {Path(project_path).name}\n\nUnable to generate README: {str(e)[:100]}"


def save_readme(project_path: str, output_path: Optional[str] = None) -> str:
    """Save README.md to disk."""
    readme_content = generate_project_readme(project_path)
    
    if not output_path:
        output_path = Path(project_path) / "README.md"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(readme_content, encoding='utf-8')
    return str(output_path)