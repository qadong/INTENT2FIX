import logging
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node

from INTENT2FIX.codeblocks.codeblocks import (
    CodeBlock,
    CodeBlockType,
    ReferenceScope,
    RelationshipType,
    ValidationError,
)
from INTENT2FIX.codeblocks.parser.parser import (
    CodeParser,
    NodeMatch,
    commented_out_keywords
)

logger = logging.getLogger(__name__)

class CppParser(CodeParser):

    def __init__(self, **kwargs):
        language = Language(tscpp.language())
        super().__init__(language, **kwargs)
        
        self.queries = []
        self.queries.extend(self._build_queries("cpp2.scm"))
        
        if self.apply_gpt_tweaks:
            self.gpt_queries.extend(self._build_queries("cpp_gpt.scm"))

    def find_in_tree(self, node: Node) -> NodeMatch:
        def get_first_child(n):
            """获取第一个有意义的子节点，跳过语法符号"""
            skip_types = ["{", "}", ";", "(", ")", "<", ">", ":", "[", "]", "template", "comment"]
            for child in n.children:
                if child.type not in skip_types:
                    return child
            return None

        def get_last_child(n):
            """获取最后一个有意义的子节点"""
            skip_types = ["{", "}", ";", "(", ")", "<", ">", ":", "[", "]", "template", "comment"]
            for child in reversed(n.children):
                if child.type not in skip_types:
                    return child
            return None

        # 1. 处理模板声明
        if node.type == "template_declaration":
            declaration = next(
                (c for c in node.children 
                if c.type in ["class_specifier", "struct_specifier", "function_definition"]),
                None
            )
            
            if declaration:
                # 提取模板参数
                template_params = []
                params_node = next(
                    (c for c in node.children if c.type == "template_parameter_list"),
                    None
                )
                if params_node:
                    for param in params_node.children:
                        if param.type == "type_parameter_declaration":
                            type_id = next(
                                (c for c in param.children if c.type == "type_identifier"),
                                None
                            )
                            if type_id:
                                template_params.append(type_id.text.decode())

                # 处理模板类/结构体
                if declaration.type in ["class_specifier", "struct_specifier"]:
                    name_node = declaration.child_by_field_name("name")
                    block_type = (CodeBlockType.TEMPLATE_CLASS if declaration.type == "class_specifier" 
                                else CodeBlockType.TEMPLATE_STRUCT)
                    
                    return NodeMatch(
                        block_type=block_type,
                        identifier_node=name_node,
                        first_child=get_first_child(declaration),
                        last_child=get_last_child(declaration),
                        metadata={
                            "is_template": True,
                            "template_parameters": template_params
                        }
                    )
                
                # 处理模板函数
                elif declaration.type == "function_definition":
                    declarator = declaration.child_by_field_name("declarator")
                    name_node = (declarator.child_by_field_name("declarator") 
                                if declarator else None)
                    
                    return NodeMatch(
                        block_type=CodeBlockType.TEMPLATE_FUNCTION,
                        identifier_node=name_node,
                        first_child=get_first_child(declaration),
                        last_child=get_last_child(declaration),
                        metadata={
                            "is_template": True,
                            "template_parameters": template_params
                        }
                    )

        # 2. 处理类和结构体定义
        if node.type in ["class_specifier", "struct_specifier"]:
            name_node = node.child_by_field_name("name")
            is_template = node.parent and node.parent.type == "template_declaration"
            
            # 使用您枚举中定义的准确类型
            if node.type == "class_specifier":
                block_type = CodeBlockType.TEMPLATE_CLASS if is_template else CodeBlockType.CLASS
            else:  # struct_specifier
                block_type = CodeBlockType.TEMPLATE_STRUCT if is_template else CodeBlockType.STRUCT
            
            return NodeMatch(
                block_type=block_type,
                identifier_node=name_node,
                first_child=get_first_child(node),
                last_child=get_last_child(node),
                metadata={"is_template": is_template} if is_template else None
            )

        # 3. 处理函数定义
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if not declarator:
                return NodeMatch(
                    block_type=CodeBlockType.FUNCTION,
                    first_child=get_first_child(node),
                    last_child=get_last_child(node)
                )
                
            name_node = declarator.child_by_field_name("declarator")
            func_name = name_node.text.decode() if name_node else ""
            
            # 处理析构函数
            if (name_node and name_node.type == "destructor_name") or func_name.startswith("~"):
                return NodeMatch(
                    block_type=CodeBlockType.DESTRUCTOR,
                    identifier_node=name_node,
                    first_child=get_first_child(node),
                    last_child=get_last_child(node)
                )
            
            # 处理构造函数
            parent = node.parent
            if parent and parent.type in ["class_specifier", "struct_specifier"]:
                class_name_node = parent.child_by_field_name("name")
                if (class_name_node and func_name == class_name_node.text.decode()):
                    return NodeMatch(
                        block_type=CodeBlockType.CONSTRUCTOR,
                        identifier_node=name_node,
                        first_child=get_first_child(node),
                        last_child=get_last_child(node)
                    )
            
            # 处理普通函数/方法
            is_method = node.parent and node.parent.type in ["class_specifier", "struct_specifier"]
            return NodeMatch(
                block_type=CodeBlockType.METHOD if is_method else CodeBlockType.FUNCTION,
                identifier_node=name_node,
                first_child=get_first_child(node),
                last_child=get_last_child(node)
            )

        # 4. 处理成员变量
        if node.type == "field_declaration":
            declarator = node.child_by_field_name("declarator")
            name_node = declarator.child_by_field_name("declarator") if declarator else None
            return NodeMatch(
                block_type=CodeBlockType.MEMBER,
                identifier_node=name_node
            )

        # 5. 处理命名空间
        if node.type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body")
            return NodeMatch(
                block_type=CodeBlockType.NAMESPACE,
                identifier_node=name_node,
                first_child=get_first_child(body_node) if body_node else None,
                last_child=get_last_child(body_node) if body_node else None
            )

        # 6. 处理lambda表达式
        if node.type == "lambda_expression":
            return NodeMatch(
                block_type=CodeBlockType.LAMBDA,
                first_child=get_first_child(node),
                last_child=get_last_child(node)
            )
            
        if node.type == "comment":
            return NodeMatch(
                block_type=CodeBlockType.COMMENT,
            )

        # 7. 处理访问修饰符
        if node.type == "access_specifier":
            return NodeMatch(
                block_type=CodeBlockType.ACCESS_MODIFIER,
                identifier_node=node
            )

        # 8. 默认返回基础代码块
        return NodeMatch(
            block_type=CodeBlockType.CODE,
            first_child=get_first_child(node),
            last_child=get_last_child(node)
        )

    def _build_code_blocks(self, node: Node, code: str, parent_block: CodeBlock = None) -> CodeBlock:
        """递归构建C++代码块结构，改进嵌套处理"""
        node_match = self.find_in_tree(node)
        
        # 跳过不需要生成代码块的节点类型
        skip_types = ["{", "}", ";", "(", ")", "<", ">", ":", "[", "]"]
        if node.type in skip_types:
            return None

        content = code[node.start_byte:node.end_byte]
        
        # 创建代码块
        code_block = CodeBlock(
            type=node_match.block_type,
            identifier=code[node_match.identifier_node.start_byte:node_match.identifier_node.end_byte].decode() 
                    if node_match.identifier_node else None,
            content=content,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent=parent_block
        )
        
        # 处理关系和参数
        # if self._enable_code_graph:
        #     code_block.relationships = self._create_references(code, node_match)
        #     code_block.parameters = self._create_parameters(code, node_match)


        code_block.relationships = self._create_references(code, node_match)
        code_block.parameters = self._create_parameters(code, node_match)
    
        print('CODEBLOCK:',code_block)


        # 预处理
        self._pre_process_block(code_block, node_match)
        
        # 递归处理子节点（添加空值检查）
        current_node = node_match.first_child
        while current_node:
            if current_node == node_match.last_child:
                break
                
            child_block = self._build_code_blocks(current_node, code, code_block)
            if child_block:  # 新增空值检查
                code_block.append_child(child_block)
            current_node = current_node.next_sibling
        
        # 处理最后一个子节点（添加空值检查）
        if node_match.last_child and node_match.last_child != node_match.first_child:
            child_block = self._build_code_blocks(node_match.last_child, code, code_block)
            if child_block:  # 新增空值检查
                code_block.append_child(child_block)
        
        # 后处理
        self._post_process_block(code_block)
        return code_block

    def _pre_process_block(self, codeblock: CodeBlock, node_match: NodeMatch):
        """C++特定的预处理逻辑"""
        # 处理构造函数和析构函数
        if codeblock.type == CodeBlockType.FUNCTION:
            if codeblock.identifier and codeblock.identifier.startswith("~"):
                codeblock.type = CodeBlockType.DESTRUCTOR
            elif (node_match.identifier_node and 
                  codeblock.parent and 
                  codeblock.parent.type in [CodeBlockType.CLASS, CodeBlockType.STRUCT] and
                  codeblock.identifier == codeblock.parent.identifier):
                codeblock.type = CodeBlockType.CONSTRUCTOR
        
        # 处理模板类
        if hasattr(node_match, 'metadata') and node_match.metadata.get("template_params"):
            if codeblock.type == CodeBlockType.CLASS:
                codeblock.type = CodeBlockType.TEMPLATE_CLASS
            elif codeblock.type == CodeBlockType.STRUCT:
                codeblock.type = CodeBlockType.TEMPLATE_STRUCT

    def _post_process_block(self, codeblock: CodeBlock):
        """C++特定的后处理逻辑"""
        # 处理注释掉的代码
        if (codeblock.type == CodeBlockType.COMMENT and 
            self._is_commented_out_code(codeblock.content)):
            codeblock.type = CodeBlockType.COMMENTED_OUT_CODE
            
        # 验证重复定义
        if codeblock.type in [CodeBlockType.CLASS, CodeBlockType.FUNCTION, CodeBlockType.STRUCT]:
            siblings = [b for b in codeblock.parent.children if b.type == codeblock.type]
            if len(siblings) > 1:
                codeblock.validation_errors.append(
                    ValidationError(error=f"Duplicate {codeblock.type} definition: {codeblock.identifier}")
                )

    def _create_references(self, code: str, node_match: NodeMatch) -> list:
        """创建C++特定的引用关系"""
        references = []
        for ref_node, ref_type in node_match.relationships:
            ref_content = code[ref_node.start_byte:ref_node.end_byte].decode()
            
            if ref_type == "reference.type":
                rel_type = RelationshipType.IS_A
            elif ref_type == "reference.call":
                rel_type = RelationshipType.CALLS
            else:
                rel_type = RelationshipType.USES
                
            references.append({
                "type": rel_type,
                "scope": ReferenceScope.LOCAL,
                "path": ref_content.split("::")
            })
        return references

    def _create_parameters(self, code: str, node_match: NodeMatch) -> list:
        """创建C++特定的参数列表"""
        parameters = []
        for param_node, type_node in node_match.parameters:
            param_name = code[param_node.start_byte:param_node.end_byte].decode()
            param_type = code[type_node.start_byte:type_node.end_byte].decode() if type_node else None
            parameters.append({
                "identifier": param_name,
                "type": param_type
            })
        return parameters

    def _is_commented_out_code(self, comment: str) -> bool:
        """检查是否是注释掉的代码"""
        return (comment.startswith("/*...") or 
                any(keyword in comment.lower() for keyword in commented_out_keywords))