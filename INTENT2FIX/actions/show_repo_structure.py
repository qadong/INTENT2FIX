import os
import subprocess
from pathlib import Path
from tree_sitter import Language, Parser
from typing import List, Dict, Tuple, Optional, Union

class SourceCodeParser:
    def __init__(self, query_dir: Union[str, Path] = None):
        self.library_dir = Path.home() / ".tree-sitter" / "bin"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_dir = Path(query_dir) if query_dir else None
        
        self.languages = {
            'c': {
                'repo': 'https://github.com/tree-sitter/tree-sitter-c',
                'exts': ['.c', '.h'],
                'so': self.library_dir / 'c.so',
                'name': 'c',
                'query_path': None  # 将在_load_queries中确定
            },
            'cpp': {
                'repo': 'https://github.com/tree-sitter/tree-sitter-cpp',
                'exts': ['.cpp', '.cc', '.cxx', '.hpp', '.h'],
                'so': self.library_dir / 'cpp.so',
                'name': 'cpp',
                'query_path': None
            },
            'java': {
                'repo': 'https://github.com/tree-sitter/tree-sitter-java',
                'exts': ['.java'],
                'so': self.library_dir / 'java.so',
                'name': 'java',
                'query_path': None
            }
        }
        
        self.default_queries = {
            'cpp': """
            ;; 默认C++查询
            (class_specifier) @class.def
            (function_definition) @function.def
            (struct_specifier) @struct.def
            """,
            'c': """
            ;; 默认C查询
            (function_definition) @function.def
            (struct_specifier) @struct.def
            """,
            'java': """
            ;; 默认Java查询
            (class_declaration) @class.def
            (method_declaration) @method.def
            """
        }
        
        self._compile_languages()
        self._load_queries()
        self._load_languages()
        
        self.parser = Parser()

    def _compile_languages(self):
        for lang, config in self.languages.items():
            if not config['so'].exists():
                self._compile_language(lang, config['repo'], config['so'])

    def _compile_language(self, lang: str, repo_url: str, output_path: Path):
        print(f"🛠️ Compiling {lang} parser...")
        
        repo_dir = self.library_dir / f"tree-sitter-{lang}"
        
        if not repo_dir.exists():
            subprocess.run([
                'git', 'clone', '--depth', '1', 
                repo_url, str(repo_dir)
            ], check=True)
        
        src_dir = repo_dir / "src"
        src_files = [str(src_dir / "parser.c")]
        
        scanner_files = [
            src_dir / "scanner.c",
            src_dir / "scanner.cc"
        ]
        for scanner in scanner_files:
            if scanner.exists():
                src_files.append(str(scanner))
        
        compile_cmd = [
            'gcc', '-shared', '-fPIC', '-fno-exceptions', '-O2',
            '-I', str(src_dir),
            '-o', str(output_path),
            *src_files
        ]
        
        try:
            subprocess.run(compile_cmd, check=True)
            print(f"✅ {lang} parser compiled successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to compile {lang} parser: {str(e)}\n"
                f"Command: {' '.join(compile_cmd)}"
            )

    def _load_queries(self):
        for lang, config in self.languages.items():
            query_text = ""
            query_path = None
            
            if self.query_dir:
                custom_path = self.query_dir / f"{lang}.scm"
                if custom_path.exists():
                    try:
                        query_text = custom_path.read_text(encoding='utf-8')
                        query_path = custom_path
                        print(f"✅ Loaded {lang} queries from {custom_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to read {custom_path}: {str(e)}")
            
            if not query_text:
                query_text = self.default_queries.get(lang, "")
                print(f"ℹ️ Using default queries for {lang}")
            
            config['query_text'] = query_text
            config['query_path'] = query_path

    def _load_languages(self):
        for lang, config in self.languages.items():
            try:
                lib_path = str(config['so'].resolve())
                config['language'] = Language(lib_path, config['name'])
                
                if config['query_text']:
                    try:
                        config['query'] = config['language'].query(config['query_text'])
                        print(f"✅ Compiled {lang} queries successfully")
                    except Exception as e:
                        if config['query_path']:
                            error_path = config['query_path'].parent / f"{lang}_ERROR.scm"
                            error_path.write_text(config['query_text'])
                            print(f"❌ Query syntax error in {config['query_path']}")
                            print(f"    See broken query at: {error_path}")
                        raise
                else:
                    config['query'] = None
                    print(f"⚠️ No queries available for {lang}")
                    
            except Exception as e:
                print(f"❌ Failed to initialize {lang}: {str(e)}")
                config['language'] = None
                config['query'] = None

    def parse_file(self, file_path: Union[str, Path]) -> Tuple[List[Dict], List[Dict], List[str]]:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ File not found: {path}")
            return [], [], []
        
        lang = self._detect_language(path)
        if not lang or not self.languages[lang]['query']:
            return [], [], []
        
        try:
            with open(path, 'rb') as f:
                content = f.read().decode('utf-8')
        except Exception as e:
            print(f"⛔ Failed to read {path}: {str(e)}")
            return [], [], []
        
        self.parser.set_language(self.languages[lang]['language'])
        tree = self.parser.parse(bytes(content, 'utf-8'))
        
        return self._query_ast(tree, lang, content.splitlines())

    def _detect_language(self, file_path: Path) -> Optional[str]:
        ext = file_path.suffix.lower()
        for lang, config in self.languages.items():
            if ext in config['exts']:
                return lang
        return None

    def _query_ast(self, tree, lang: str, lines: List[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
        classes = []
        functions = []
        
        query = self.languages[lang]['query']
        if not query:
            return [], [], lines
        
        for node, tag in query.captures(tree.root_node):
            if tag == 'class.def' or tag == 'struct.def':
                name = self._get_node_name(node, ['identifier', 'type_identifier'])
                classes.append({
                    'name': name or 'anonymous',
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'text': lines[node.start_point[0]:node.end_point[0]+1],
                    'methods': self._extract_methods(node, lines)
                })
            elif tag == 'function.def':
                name = self._get_node_name(node, ['identifier', 'field_identifier'])
                functions.append({
                    'name': name or 'anonymous',
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'text': lines[node.start_point[0]:node.end_point[0]+1]
                })
        
        return classes, functions, lines

    @staticmethod
    def _get_node_name(node, possible_types: List[str]) -> Optional[str]:
        for child in node.children:
            if child.type in possible_types:
                return child.text.decode('utf-8')
        return None

    @staticmethod
    def _extract_methods(class_node, lines: List[str]) -> List[Dict]:
        methods = []
        for child in class_node.children:
            if child.type == 'function_definition':
                name = SourceCodeParser._get_node_name(child, ['identifier', 'field_identifier'])
                if name:
                    methods.append({
                        'name': name,
                        'start_line': child.start_point[0] + 1,
                        'end_line': child.end_point[0] + 1,
                        'text': lines[child.start_point[0]:child.end_point[0]+1]
                    })
        return methods

    def debug_ast(self, file_path: Union[str, Path]):
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {path}")
            return
        
        lang = self._detect_language(path)
        if not lang:
            print(f"Unsupported file type: {path.suffix}")
            return
        
        with open(path, 'rb') as f:
            code = f.read()
        
        self.parser.set_language(self.languages[lang]['language'])
        tree = self.parser.parse(code)
        
        print(f"\n🌳 AST for {path.name}:")
        print(tree.root_node.sexp())
        
        if self.languages[lang]['query']:
            print("\n🔍 Query captures:")
            for node, tag in self.languages[lang]['query'].captures(tree.root_node):
                print(f"{tag}: {node.text.decode('utf-8')[:50]}... (lines {node.start_point[0]+1}-{node.end_point[0]+1})")

def analyze_project(project_path: str) -> Dict:
    current_dir = Path(__file__).parent.parent  # INTENT2FIX/
    query_dir = current_dir / "codeblocks" / "parser" / "queries"
    parser = SourceCodeParser(
        query_dir=str(query_dir)
    )
    structure = {}
    
    try:
        project_path = Path(project_path).resolve()
        if not project_path.exists():
            raise ValueError("repo dir not exist")
        
        for root, _, files in os.walk(project_path):
            # 构建相对路径
            rel_path = os.path.relpath(root, project_path)
            if rel_path == ".":
                rel_path = project_path.name
            
            # 创建嵌套字典结构
            current_level = structure
            for part in rel_path.split(os.sep):
                current_level = current_level.setdefault(part, {})
            
            # 解析文件
            for filename in files:
                file_path = Path(root) / filename
                classes, funcs, content = parser.parse_file(str(file_path))
                current_level[filename] = {
                    'classes': classes,
                    'functions': funcs,
                    'text': content
                }
                
    except Exception as e:
        print(f"🔥 error: {str(e)}")
        return {}
    
    return structure


def show_project_structure(structure, spacing=0) -> str:
    """Pretty-print project structure (supports Python, C, C++, Java)"""
    pp_string = ""
    allowed_extensions = {'.py', '.c', '.cpp', '.h', '.hpp', '.java'}

    for key, value in structure.items():
        # Skip non-source files (no extension or wrong extension)
        if "." in key:
            ext = key[key.rfind('.'):].lower()
            if ext not in allowed_extensions:
                continue
            pp_string += " " * spacing + key + "\n"
        else:
            pp_string += " " * spacing + key + "/\n"
        
        # Recurse if not a file with classes
        if isinstance(value, dict) and "classes" not in value:
            pp_string += show_project_structure(value, spacing + 4)

    return pp_string

def show_code_structure(
    structure, 
    spacing=0,
    show_extensions={'.py'},  
    show_details=True  
) -> str:

    pp_string = ""
    
    for key, value in structure.items():
        # 处理目录
        if isinstance(value, dict) and not any(k in value for k in ["classes", "functions", "structs"]):
            pp_string += " " * spacing + key + "/\n"
            pp_string += show_code_structure(value, spacing + 4, show_extensions, show_details)
            
        # 处理代码文件
        elif isinstance(value, dict) and ("classes" in value or "functions" in value):
            ext = '.' + key.split('.')[-1].lower()
            
            if ext in show_extensions:
                pp_string += " " * spacing + key + "\n"
                
                if show_details:
                    # 显示类信息
                    for cls in value.get("classes", []):
                        pp_string += " " * (spacing + 4) + f"class {cls['name']}\n"
                        for method in cls.get("methods", []):
                            pp_string += " " * (spacing + 8) + f"def {method['name']}()\n"
                    
                    # 显示独立函数
                    for func in value.get("functions", []):
                        pp_string += " " * (spacing + 4) + f"def {func['name']}()\n"
                        
        # 其他文件（可选显示）
        else:
            if show_details:  # 如果要显示所有文件列表
                pp_string += " " * spacing + key + "\n"
                
    return pp_string

if __name__ == "__main__":
    project_structure = analyze_project("/tmp/INTENT2FIX/tmp/repos/swe-bench_jarro2783__cxxopts-360/src")
    
    print(project_structure)
    
    print(show_code_structure(project_structure,
                       show_extensions={'.cpp'}))