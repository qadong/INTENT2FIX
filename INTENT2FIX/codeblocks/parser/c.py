import logging
from .cpp import CppParser

logger = logging.getLogger(__name__)

class CParser(CppParser):
    @property
    def language(self):
        return "c"
