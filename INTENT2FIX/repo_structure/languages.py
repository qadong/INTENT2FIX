from tree_sitter import Language

# 语言配置
LANGUAGE_CONFIG = {
    'python': {
        'extensions': ['.py'],
        'language_name': 'python'
    },
    'c': {
        'extensions': ['.c', '.h'],
        'language_name': 'c'
    },
    'cpp': {
        'extensions': ['.cpp', '.hpp', '.cc', '.cxx', '.h'],
        'language_name': 'cpp'
    },
    'java': {
        'extensions': ['.java'],
        'language_name': 'java'
    }
}

def build_language_library(lib_path='build/languages.so'):
    """编译语言库"""
    Language.build_library(
        lib_path,
        [f'tree-sitter-{lang}' for lang in LANGUAGE_CONFIG]
    )
    return lib_path

def get_language_by_ext(ext):
    """根据扩展名获取语言类型"""
    for lang, config in LANGUAGE_CONFIG.items():
        if ext.lower() in config['extensions']:
            return lang
    return None