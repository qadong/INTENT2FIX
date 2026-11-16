import os
from tree_sitter import Parser, Node, Language
import tree_sitter_python
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_c
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import re


CPP_TREE_EXTENSIONS = {
    '.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.inl', '.ipp', '.tpp',
    '.cmake', '.bzl',
}
CPP_TREE_FILES = {
    'CMakeLists.txt', 'CMakePresets.json', 'CMakeUserPresets.json', 'Makefile',
    'meson.build', 'meson_options.txt', 'BUILD', 'BUILD.bazel', 'WORKSPACE',
    'WORKSPACE.bazel', 'configure', 'configure.ac',
}
CPP_TREE_IGNORED_DIRS = {
    '.git', '.hg', '.svn', '.cache', '.idea', '.vscode', '.venv', '__pycache__',
    '_build', 'build', 'dist', 'node_modules', 'out', 'venv',
}


def show_cpp_repo_structure(directory_path: str) -> str:
    """Render a navigation-focused C++ file tree without parsing source bodies."""
    paths = []
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = sorted(
            directory for directory in dirs
            if directory not in CPP_TREE_IGNORED_DIRS
            and not directory.startswith('cmake-build-')
        )
        for filename in sorted(files):
            suffix = os.path.splitext(filename)[1].lower()
            if filename in CPP_TREE_FILES or suffix in CPP_TREE_EXTENSIONS:
                relative_path = os.path.relpath(os.path.join(root, filename), directory_path)
                paths.append(relative_path.split(os.sep))

    tree = {}
    for parts in paths:
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None

    lines = []

    def render(node: Dict, depth: int) -> None:
        for name, child in sorted(node.items(), key=lambda item: (item[1] is None, item[0].lower())):
            indent = "    " * depth
            if child is None:
                lines.append(f"{indent}{name}")
            else:
                lines.append(f"{indent}{name}/")
                render(child, depth + 1)

    render(tree, 0)
    return "\n".join(lines)

class CodeParser:
    def __init__(self):
        # 加载各语言对象，并用 Language 包装
        self._languages = {
            'python': Language(tree_sitter_python.language()),
            'c': Language(tree_sitter_c.language()),
            'cpp': Language(tree_sitter_cpp.language()),
            'java': Language(tree_sitter_java.language())
        }
        self._parsers = self._init_parsers()

    def _init_parsers(self) -> Dict[str, Parser]:
        """初始化 Parser 实例，并设置对应的语言"""
        return {
            lang: Parser(lang_obj)  # 直接传入 Language 对象
            for lang, lang_obj in self._languages.items()
        }

    
    def get_language_parser(self, file_ext: str) -> Optional[Parser]:
        """根据文件扩展名获取解析器"""
        ext_mapping = {
            '.py': 'python',
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.hpp': 'cpp',
            '.cc': 'cpp', '.cxx': 'cpp',
            '.java': 'java'
        }
        return self._parsers.get(ext_mapping.get(file_ext.lower()))
    
    def parse_file(self, file_path: str, file_content: str = None) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        解析代码文件
        :param file_path: 文件路径
        :param file_content: 可选的文件内容
        :return: (classes, functions, lines)
        """
        ext = os.path.splitext(file_path)[1]
        parser = self.get_language_parser(ext)
        
        if not parser:
            return [], [], []
        
        try:
            code = file_content.encode('utf-8') if file_content else open(file_path, 'rb').read()
            tree = parser.parse(code)
            code_text = code.decode('utf-8')
            return self._extract_ast_info(tree.root_node, code_text)
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return [], [], []
    
    def _extract_ast_info(self, root_node: Node, code_text: str) -> Tuple[List[Dict], List[Dict], List[str]]:
        """从AST节点提取信息"""
        classes = []
        functions = []
        
        def _walk(node: Node):
            # 处理类定义
            if node.type in ('class_definition', 'class_specifier', 'class_declaration'):
                class_info = self._extract_class_info(node, code_text)
                if class_info:
                    classes.append(class_info)
            
            # 处理函数定义
            elif node.type in ('function_definition', 'method_declaration', 'function_declaration'):
                func_info = self._extract_function_info(node, code_text)
                if func_info:
                    functions.append(func_info)
            
            for child in node.children:
                _walk(child)
        
        _walk(root_node)
        return classes, functions, code_text.splitlines()
    
    def _extract_class_info(self, node: Node, code_text: str) -> Optional[Dict[str, Any]]:
        """提取类信息"""
        name_node = self._find_child_by_type(node, ['identifier', 'type_identifier'])
        if not name_node:
            return None
            
        class_name = code_text[name_node.start_byte:name_node.end_byte].strip()
        if not class_name or len(class_name) < 2:
            return None
            
        return {
            'name': class_name,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'type': node.type
        }
    
    def _extract_function_info(self, node: Node, code_text: str) -> Optional[Dict[str, Any]]:
            """改进版函数信息提取"""
            name_node = self._find_function_name_node(node)
            if not name_node:
                return None
                
            func_name = code_text[name_node.start_byte:name_node.end_byte].strip()
            
            # 增加名称过滤逻辑
            if not self._is_valid_function_name(func_name):
                return None
                
            return {
                'name': func_name,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'type': node.type
            }

    @staticmethod
    def _is_valid_function_name(name: str) -> bool:
        """验证函数名有效性"""
        return re.fullmatch(r'[a-zA-Z_][\w:]*(<.*>)?', name) is not None  # 支持模板函数
    def _find_function_name_node(self, node: Node) -> Optional[Node]:
        """改进版函数名查找（支持复杂声明）"""
        if node.type == 'function_definition':
            # C++函数需要处理更复杂的声明结构
            declarator = self._find_child_by_type(node, ['function_declarator'])
            if declarator:
                # 查找声明中的直接identifier或通过pointer_declarator查找
                if name_node := self._find_deep_child_by_type(declarator, ['identifier']):
                    return name_node
                # 处理函数指针等复杂情况
                if ptr_declarator := self._find_deep_child_by_type(declarator, ['pointer_declarator']):
                    return self._find_deep_child_by_type(ptr_declarator, ['identifier'])
        elif node.type == 'method_declaration':
            return self._find_deep_child_by_type(node, ['identifier'])
        return None

    def _find_deep_child_by_type(self, node: Node, target_types: List[str]) -> Optional[Node]:
        """增强版深度查找（带过滤条件）"""
        for child in node.children:
            if child.type in target_types:
                # 验证标识符有效性
                if self._is_valid_identifier(child):
                    return child
            if found := self._find_deep_child_by_type(child, target_types):
                return found
        return None

    @staticmethod
    def _is_valid_identifier(node: Node) -> bool:
        """验证是否为合法标识符（防止捕获特殊符号）"""
        text = node.text.decode('utf-8')  # 假设有访问text的方法
        return re.match(r'^[a-zA-Z_]\w*$', text) is not None
    @staticmethod
    def _find_child_by_type(node: Node, target_types: List[str]) -> Optional[Node]:
        """查找指定类型的子节点"""
        for child in node.children:
            if child.type in target_types:
                return child
        return None


class DirectoryAnalyzer:
    """目录结构分析器（优化版）"""
    
    def __init__(self, code_parser: CodeParser):
        self.parser = code_parser
    
    def analyze(self, directory_path: str) -> Dict:
        """
        分析代码目录结构
        :param directory_path: 目录路径
        :return: 嵌套字典结构
        """
        structure = {}
        
        for root, _, files in os.walk(directory_path):
            current_level = self._get_nested_level(structure, root, directory_path)
            
            for file in tqdm(files, desc=f"Parsing {os.path.relpath(root, directory_path) or 'root'}"):
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                
                if self.parser.get_language_parser(ext):
                    classes, functions, lines = self.parser.parse_file(file_path)
                    current_level[file] = {
                        'classes': classes,
                        'functions': functions,
                        'line_count': len(lines),
                        'language': self._get_language_name(ext)
                    }
        
        return structure
    
    @staticmethod
    def _get_nested_level(structure: Dict, root: str, base_path: str) -> Dict:
        """获取嵌套层级"""
        rel_path = os.path.relpath(root, base_path)
        if rel_path == '.':
            return structure
        
        current = structure
        for part in rel_path.split(os.sep):
            current = current.setdefault(part, {})
        return current
    
    @staticmethod
    def _get_language_name(ext: str) -> str:
        """获取语言名称"""
        lang_map = {
            '.py': 'python',
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.hpp': 'cpp',
            '.cc': 'cpp', '.cxx': 'cpp',
            '.java': 'java'
        }
        return lang_map.get(ext.lower(), 'unknown')


def show_code_structure(
    structure, 
    spacing=0,
    show_extensions={'.hpp', '.cpp', '.h', '.c', '.py','.cc', '.java','.cxx'},  # 新增扩展名集合
    show_details=False,
    ignore_dirs={'.github', '.git'}  # 新增忽略目录集合
) -> str:
    """过滤.git和.github的类结构展示"""
    pp_string = ""
    
    for key, value in structure.items():
        # 跳过忽略目录
        if key in ignore_dirs:
            continue
            
        # 处理目录
        if isinstance(value, dict) and not any(k in value for k in ["classes"]):
            pp_string += " " * spacing + f" {key}/\n"
            pp_string += show_code_structure(value, spacing + 4, show_extensions, show_details, ignore_dirs)
            
        # 处理代码文件
        elif isinstance(value, dict) and "classes" in value:
            ext = '.' + key.split('.')[-1].lower()
            
            if ext in show_extensions:
                pp_string += " " * spacing + f" {key}\n"
                
                if show_details:
                    for cls in value.get("classes", []):
                        cls_name = cls['name'].strip()
                        if (
                            len(cls_name) > 2 
                            and not cls_name.startswith(('{', ':', ' '))
                            and '\n' not in cls_name
                        ):
                            pp_string += (
                                " " * (spacing + 4) 
                                + f"class {cls_name} (lines {cls['start_line']}-{cls['end_line']})\n"
                            )

    return pp_string
