"""CodeIndex implementation using entity_tools as the search backend"""

import logging
import fnmatch
from pathlib import PurePosixPath
from pathlib import Path
from typing import Optional, Dict, Any, List

from INTENT2FIX.index.runtime_preflight import validate_runtime_index
from INTENT2FIX.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
from INTENT2FIX.repository.repository import Repository

# Import entity_tools
try:
    from code_intent.code_intent.entity_tools import (
        search_symbol,
        search_by_intent,
        get_entity_by_id,
        find_template_instantiation,
        list_features,
        set_current_instance,
        set_current_index_store,
    )
except ImportError:
    # Fallback for testing
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from code_intent.code_intent.entity_tools import (
            search_symbol,
            search_by_intent,
            get_entity_by_id,
            find_template_instantiation,
            list_features,
            set_current_instance,
            set_current_index_store,
        )
    except ImportError:
        search_symbol = None
        search_by_intent = None
        get_entity_by_id = None
        find_template_instantiation = None
        list_features = None
        set_current_instance = None
        set_current_index_store = None

logger = logging.getLogger(__name__)


def _matches_file_pattern(file_path: str, file_pattern: str) -> bool:
    path = file_path.replace("\\", "/").lstrip("./")
    pattern = file_pattern.replace("\\", "/").lstrip("./")
    if not path or not pattern:
        return False
    if any(char in pattern for char in "*?["):
        patterns = {pattern}
        while "**/" in pattern:
            pattern = pattern.replace("**/", "", 1)
            patterns.add(pattern)
        return any(
            fnmatch.fnmatch(path, candidate)
            or PurePosixPath(path).match(candidate)
            for candidate in patterns
        )
    return path == pattern or path.startswith(pattern.rstrip("/") + "/")


def _generate_span_id(file_path: str, start_line: int, end_line: int) -> str:
    """
    Generate span_id using file_path and line range.
    Format: {file_path}:{start_line} or {file_path}:{start_line}-{end_line}
    
    Example:
        - "src/catch2/catch_test_macros.hpp:126"
        - "src/catch2/catch_test_macros.hpp:126-150"
    
    This format allows the agent to track which code spans have been viewed
    using a direct file:line identifier.
    """
    if start_line == end_line:
        return f"{file_path}:{start_line}"
    return f"{file_path}:{start_line}-{end_line}"


def _entity_to_search_hit(
    entity: Dict[str, Any],
    repo: Optional[Repository] = None,
    skeleton: bool = False,
) -> Optional[SearchCodeHit]:
    """
    Convert entity result to SearchCodeHit.
    
    Args:
        entity: Entity dictionary from entity_tools
        repo: Repository for path normalization
        skeleton: If True, only include the start line (signature) of the entity
    
    Returns:
        SearchCodeHit or None if conversion fails
    """
    # Get file path
    file_path = entity.get("file") or entity.get("def_file")
    if not file_path:
        logger.warning(f"Entity missing file path: {entity.get('id')}")
        return None
    
    # Normalize path using repository
    if repo and hasattr(repo, "get_relative_path"):
        try:
            file_path = repo.get_relative_path(file_path)
        except Exception:
            # If path normalization fails, use original path
            pass
    
    # Get line range
    line_range = entity.get("line_range")
    if line_range and len(line_range) >= 2:
        start_line, end_line = line_range[0], line_range[1]
    else:
        # Fallback to various line fields used by entity_tools or RAG
        start_line = (
            entity.get("def_start_line") or 
            entity.get("def_line") or 
            entity.get("line") or 
            entity.get("trigger_line") or 
            entity.get("start_line") or
            1
        )
        end_line = entity.get("def_end_line") or entity.get("end_line") or start_line

    # If skeleton mode, only show the signature (first line)
    if skeleton:
        end_line = start_line

    # Handle 0-based line numbers from some parsers
    if start_line == 0:
        start_line = 1
    if end_line == 0:
        end_line = 1
    
    # Handle -1 or other invalid end lines
    if end_line < start_line:
        end_line = start_line
    
    # Get fq_name for span_id generation
    fq_name = entity.get("fq_name")
    if not fq_name:
        # Fallback to name or id
        fq_name = entity.get("name") or entity.get("id", "unknown")
    
    # Generate span_id: file_path:start_line
    span_id = _generate_span_id(file_path, start_line, end_line)
    
    # Estimate tokens (rough estimate: ~4 tokens per line)
    # Ensure tokens is at least 0
    tokens = max(0, (end_line - start_line + 1)) * 4
    
    logger.debug(f"Converted entity {fq_name} to hit: {file_path}:{start_line}-{end_line} (span_id={span_id})")
    
    # Create SpanHit with line information
    # The span_id format fq_name@{start_line}-{end_line} contains both start and end lines
    # which can be parsed by FileContext.add_span()
    span_hit = SpanHit(
        span_id=span_id,
        label=fq_name,
        start_line=start_line,
        end_line=end_line,
        rank=0,
        tokens=tokens,
        skeleton=entity.get("skeleton")
    )
    
    return SearchCodeHit(
        file_path=file_path,
        spans=[span_hit]
    )


class EntityCodeIndex:
    """
    CodeIndex implementation that uses entity_tools as the search backend.
    This completely replaces the original CodeIndex for C++ codebases.
    
    EntityCodeIndex doesn't need to load from persist directories because
    entity_tools reads directly from JSONL files in the index directory.
    """
    
    def __init__(
        self,
        file_repo: Repository | None = None,
        index_name: Optional[str] = None,
        index_store_dir: Optional[str] = None,
        **kwargs
    ):
        import os
        self._file_repo = file_repo
        self._index_name = index_name
        configured_root = index_store_dir or os.environ.get("INDEX_STORE_DIR")
        self._index_store_dir = (
            str(Path(configured_root).expanduser().resolve()) if configured_root else None
        )
        self._activate_instance()

        self._runtime_preflight = None
        if self._index_name and self._index_store_dir:
            repo_dir = getattr(file_repo, "repo_dir", None) if file_repo else None
            self._runtime_preflight = validate_runtime_index(
                self._index_name,
                repo_dir,
                self._index_store_dir,
            )
        
        if search_symbol is None:
            logger.warning(
                "entity_tools not available. EntityCodeIndex will not work properly."
            )
        
        logger.info(
            f"Initialized EntityCodeIndex (index_name={index_name}) "
            f"using entity_tools backend. Repository: {file_repo.repo_dir if file_repo else 'None'}, "
            f"INDEX_STORE_DIR: {self._index_store_dir or 'not set'}"
        )
        if self._runtime_preflight:
            logger.info(
                "Canonical runtime index validated for %s: commit=%s files=%d",
                self._runtime_preflight["instance"],
                self._runtime_preflight["base_commit"],
                self._runtime_preflight["indexed_file_count"],
            )

    def _activate_instance(self) -> None:
        """Bind entity_tools calls to this C++ index instance."""
        if set_current_index_store is not None:
            set_current_index_store(self._index_store_dir)
        if not self._index_name:
            return
        import os

        if set_current_instance is not None:
            set_current_instance(self._index_name)
        os.environ["SWEBENCH_INSTANCE_NAME"] = self._index_name
        logger.debug(f"Set SWEBENCH_INSTANCE_NAME={self._index_name} for entity_tools")
    
    @classmethod
    def from_index_name(
        cls,
        index_name: str,
        file_repo: Repository,
        index_store_dir: Optional[str] = None,
    ) -> "EntityCodeIndex":
        """
        Create EntityCodeIndex from index name.
        
        The supplied index root is bound to the current worker thread so an
        explicit run configuration cannot silently fall back to a stale
        process-level ``INDEX_STORE_DIR``.
        
        Args:
            index_name: Instance name (used by entity_tools to find the index)
            file_repo: Repository instance
            index_store_dir: Explicit runtime index root (defaults to the environment)
        
        Returns:
            EntityCodeIndex instance
        """
        # Set the instance name for entity_tools
        import os
        if not os.environ.get("SWEBENCH_INSTANCE_NAME"):
            os.environ["SWEBENCH_INSTANCE_NAME"] = index_name
            logger.info(f"Set SWEBENCH_INSTANCE_NAME={index_name} for entity_tools")
        
        return cls(
            file_repo=file_repo,
            index_name=index_name,
            index_store_dir=index_store_dir,
        )
    
    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        file_repo: Repository | None = None,
        **kwargs
    ) -> "EntityCodeIndex":
        """
        Create EntityCodeIndex from persist directory.
        
        Note: EntityCodeIndex doesn't actually load from persist_dir.
        Instead, entity_tools reads from the index directory.
        This method is kept for compatibility with existing code.
        
        Args:
            persist_dir: Not used, kept for compatibility
            file_repo: Repository instance
            **kwargs: Additional arguments (ignored)
        
        Returns:
            EntityCodeIndex instance
        """
        # Try to infer instance name from persist_dir path
        import os
        index_name = None
        if persist_dir:
            # Extract instance name from path if possible
            # e.g., /path/to/index_store/instance_name -> instance_name
            parts = os.path.normpath(persist_dir).split(os.sep)
            if parts:
                index_name = parts[-1]
        
        index_store_dir = str(Path(persist_dir).expanduser().resolve().parent) if persist_dir else None
        return cls(
            file_repo=file_repo,
            index_name=index_name,
            index_store_dir=index_store_dir,
        )
    
    @classmethod
    def from_url(
        cls,
        url: str,
        persist_dir: str,
        file_repo: Repository,
    ) -> "EntityCodeIndex":
        """
        Create EntityCodeIndex from URL.
        
        Note: EntityCodeIndex doesn't download from URL.
        This method is kept for compatibility but doesn't download anything.
        
        Args:
            url: Not used, kept for compatibility
            persist_dir: Used to infer instance name
            file_repo: Repository instance
        
        Returns:
            EntityCodeIndex instance
        """
        return cls.from_persist_dir(persist_dir, file_repo=file_repo)
    
    def dict(self):
        """Return dictionary representation for serialization."""
        return {"index_name": self._index_name}
    
    def find_function(
        self,
        function_name: str,
        class_name: Optional[str] = None,
        file_pattern: Optional[str] = None,
        namespace: Optional[str] = None,
        name_exact_match: bool = False,
        return_type: Optional[str] = None,
        parameter_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> SearchCodeResponse:
        """
        Find function by name, including function templates.
        
        Args:
            function_name: Function name to search for
            class_name: Optional class name if searching for a method
            file_pattern: Optional file pattern filter
            namespace: Optional namespace filter
            name_exact_match: Require an exact qualified or short-name match
            return_type: Optional exact return-type filter
            parameter_types: Optional exact ordered parameter-type filter
            limit: Maximum number of unique results
        
        Returns:
            SearchCodeResponse with matching functions
        """
        if search_symbol is None:
            return SearchCodeResponse(message="entity_tools not available")
        
        try:
            self._activate_instance()
            # Search both regular functions and function templates
            types_to_search = ["function", "function_template"]
            all_entities = []
            
            for entity_type in types_to_search:
                result = search_symbol(
                    type=entity_type,
                    name=function_name,
                    name_exact_match=name_exact_match,
                    class_name=class_name,
                    file=file_pattern,
                    namespace=namespace,
                    return_type=return_type,
                    parameter_types=parameter_types,
                    limit=limit,
                )
                all_entities.extend(result.get("entities", []))
            
            # Parser/macro passes may assign different IDs to the same definition.
            unique_entities = {}
            for entity in all_entities:
                line_range = entity.get("line_range") or []
                identity = (
                    entity.get("fq_name") or entity.get("name"),
                    entity.get("file"),
                    tuple(line_range),
                    tuple(entity.get("parameters") or []),
                )
                unique_entities.setdefault(identity, entity)
            
            hits = []
            for entity in list(unique_entities.values())[:limit]:
                hit = _entity_to_search_hit(entity, self._file_repo)
                if hit:
                    hits.append(hit)
            
            # Fallback: if no hits found and class_name is provided, search for the class
            # This handles cases where methods are not indexed as separate entities (common in giant header-only classes)
            if (
                not hits
                and class_name
                and not name_exact_match
                and not (return_type or parameter_types)
            ):
                logger.info(f"No function hits for '{function_name}' in class '{class_name}'. Trying fallback to class search.")
                class_result = self.find_class(
                    class_name=class_name,
                    namespace=namespace,
                    file_pattern=file_pattern,
                    name_exact_match=name_exact_match,
                    limit=limit,
                )
                for class_hit in class_result.hits:
                    # We found the class. Since we don't have the exact function lines,
                    # we return the class hit but with a message.
                    # In a more advanced version, we could grep the class code for the function name.
                    hits.append(class_hit)
                
                if hits:
                    message = f"Function '{function_name}' not found as a separate entity, but found class '{class_name}'. The function might be defined inline within this class."
                    return SearchCodeResponse(message=message, hits=hits)

            message = f"Found {len(hits)} function(s) matching '{function_name}'"
            if class_name:
                message += f" in class '{class_name}'"
            
            return SearchCodeResponse(message=message, hits=hits)
            
        except Exception as e:
            logger.error(f"Error in find_function: {e}", exc_info=True)
            return SearchCodeResponse(
                message=f"Error searching for function: {str(e)}"
            )

    def find_class(
        self,
        class_name: str,
        file_pattern: Optional[str] = None,
        kind: Optional[str] = None,
        namespace: Optional[str] = None,
        name_exact_match: bool = False,
        limit: int = 50,
    ) -> SearchCodeResponse:
        """
        Find class by name, including class templates and structs.
        
        Args:
            class_name: Class name to search for
            file_pattern: Optional file pattern filter
            kind: Optional kind filter ('class' or 'struct')
            namespace: Optional namespace filter
            name_exact_match: Require an exact qualified or short-name match
            limit: Maximum number of unique results
        
        Returns:
            SearchCodeResponse with matching classes
        """
        if search_symbol is None:
            return SearchCodeResponse(message="entity_tools not available")
        
        try:
            self._activate_instance()
            # Search both regular classes and class templates
            types_to_search = ["class", "class_template"]
            all_entities = []
            
            for entity_type in types_to_search:
                result = search_symbol(
                    type=entity_type,
                    name=class_name,
                    name_exact_match=name_exact_match,
                    file=file_pattern,
                    kind=kind,
                    namespace=namespace,
                    limit=limit,
                )
                all_entities.extend(result.get("entities", []))
            
            unique_entities = {}
            for entity in all_entities:
                identity = (
                    entity.get("fq_name") or entity.get("name"),
                    entity.get("file"),
                    tuple(entity.get("line_range") or []),
                )
                unique_entities.setdefault(identity, entity)
            
            hits = []
            for entity in list(unique_entities.values())[:limit]:
                hit = _entity_to_search_hit(entity, self._file_repo)
                if hit:
                    hits.append(hit)
            
            message = f"Found {len(hits)} class(es) matching '{class_name}'"
            return SearchCodeResponse(message=message, hits=hits)
            
        except Exception as e:
            logger.error(f"Error in find_class: {e}", exc_info=True)
            return SearchCodeResponse(
                message=f"Error searching for class: {str(e)}"
            )
    
    def semantic_search(
        self,
        query: Optional[str] = None,
        code_snippet: Optional[str] = None,
        file_pattern: Optional[str] = None,
        category: str | None = None,
        max_results: int = 100,
        **kwargs
    ) -> SearchCodeResponse:
        """
        Semantic search for specific functions using entity_tools search_by_intent.
        """
        if search_by_intent is None:
            return SearchCodeResponse(message="entity_tools not available")
        
        if not query:
            return SearchCodeResponse(message="Query is required for semantic search")
        
        try:
            self._activate_instance()
            search_limit = 50 if file_pattern else min(max_results, 50)
            result = search_by_intent(
                query=query,
                top_k=search_limit,
            )
            
            if result.get("error"):
                logger.error(f"Error in search_by_intent: {result.get('error')}")
                # We could potentially return SearchCodeResponse(message=result.get("error")) here
                # but continuing might allow partial results if any were returned
            
            # Use a dictionary to group hits by file_path
            # file_path -> SearchCodeHit
            file_hits: Dict[str, SearchCodeHit] = {}
            
            accepted_entities = 0
            for entity in result.get("results", []):
                if file_pattern:
                    entity_file = entity.get("file", "")
                    if not _matches_file_pattern(entity_file, file_pattern):
                        continue
                
                logger.debug(
                    "semantic_search candidate: id=%s name=%s file=%s score=%s",
                    entity.get("id") or entity.get("func_id"),
                    entity.get("name") or entity.get("short_intent"),
                    entity.get("file"),
                    entity.get("score"),
                )
                hit = _entity_to_search_hit(entity, self._file_repo)

                if hit:
                    accepted_entities += 1
                    if hit.file_path not in file_hits:
                        file_hits[hit.file_path] = hit
                    else:
                        # Append the new span to existing hit for this file
                        # Avoid duplicate spans in the same file
                        existing_span_ids = {s.span_id for s in file_hits[hit.file_path].spans}
                        for new_span in hit.spans:
                            if new_span.span_id not in existing_span_ids:
                                file_hits[hit.file_path].spans.append(new_span)
                    if accepted_entities >= max_results:
                        break
            
            hits = list(file_hits.values())
            logger.info(
                "Semantic search selected %d entity span(s) across %d file(s) "
                "from %d candidate(s)%s.",
                accepted_entities,
                len(hits),
                len(result.get("results", [])),
                f" matching {file_pattern!r}" if file_pattern else "",
            )
            
            message = f"Semantic 'Function' search found {len(hits)} relevant file(s) implementing the requested functionality."
            return SearchCodeResponse(message=message, hits=hits)
            
        except Exception as e:
            logger.error(f"Error in semantic_search: {e}", exc_info=True)
            return SearchCodeResponse(
                message=f"Error in semantic search: {str(e)}"
            )

    def feature_search(
        self,
        query: Optional[str] = None,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """
        Semantic search for high-level features (functional clusters) using entity_tools list_features.
        """
        if list_features is None:
            return {"error": "entity_tools not available", "features": []}
            
        try:
            self._activate_instance()
            result = list_features(query=query, limit=max_results)
            return result
        except Exception as e:
            logger.error(f"Error in feature_search: {e}", exc_info=True)
            return {"error": str(e), "features": []}

    def find_test_files(
        self,
        file_path: str,
        span_id: str | None = None,
        query: str | None = None,
        max_results: int = 5,
        max_tokens: int | None = None,
        max_method_tokens: int = 500,
        max_spans: int | None = None,
    ) -> List[Any]:
        """
        Find test files related to the provided file path.
        
        Args:
            file_path: The file path to find tests for
            span_id: Optional span ID
            query: Optional query string
            max_results: Maximum number of files to return
            max_tokens: Maximum total tokens
            max_method_tokens: Maximum tokens per method
            max_spans: Maximum spans per file
            
        Returns:
            List of FileWithSpans
        """
        from INTENT2FIX.schema import FileWithSpans
        import os
        from rapidfuzz import fuzz
        
        logger.info(f"Finding test files for {file_path} (query: {query})")
        
        test_files = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # 1. Try to find test files by naming pattern in the repository
            if self._file_repo:
                # Find all potential test files
                all_files = self._file_repo.find_files(["**/test*/**", "*_test.*", "test_*.*"])
                
                # Scored matching based on filename similarity
                scored_tests = []
                for t_file in all_files:
                    t_base = os.path.splitext(os.path.basename(t_file))[0]
                    # Score based on how much the source filename is in the test filename
                    score = fuzz.partial_ratio(base_name.lower(), t_base.lower())
                    scored_tests.append((t_file, score))
                
                scored_tests.sort(key=lambda x: x[1], reverse=True)
                
                for t_file, score in scored_tests:
                    if score > 80:  # High confidence threshold
                        test_files.append(FileWithSpans(file_path=t_file, span_ids=[]))
                        if len(test_files) >= max_results:
                            break
            
            # 2. If no matches by name, or we need more, try semantic search
            if len(test_files) < max_results:
                search_query = f"tests for {file_path}"
                if query:
                    search_query += f" {query}"
                
                results = self.semantic_search(query=search_query, max_results=max_results * 2)
                for hit in results.hits:
                    # Avoid duplicates
                    if any(f.file_path == hit.file_path for f in test_files):
                        continue
                        
                    # Heuristic: only include if it looks like a test
                    if "test" in hit.file_path.lower() or "unit" in hit.file_path.lower():
                        span_ids = [s.span_id for s in hit.spans]
                        if max_spans:
                            span_ids = span_ids[:max_spans]
                        test_files.append(FileWithSpans(file_path=hit.file_path, span_ids=span_ids))
                        
                    if len(test_files) >= max_results:
                        break
            
            logger.info(f"Found {len(test_files)} related test files for {file_path}")
            return test_files
            
        except Exception as e:
            logger.error(f"Error in find_test_files: {e}", exc_info=True)
            return []
    
    def find_namespace(
        self,
        namespace_name: str,
        file_pattern: Optional[str] = None,
        limit: int = 50,
    ) -> SearchCodeResponse:
        """
        Find entities within a namespace to provide a 'skeleton' view.
        
        Args:
            namespace_name: Namespace name to search for
            file_pattern: Optional file pattern filter
            limit: Maximum number of unique entities
        
        Returns:
            SearchCodeResponse with entities in the namespace
        """
        if search_symbol is None:
            return SearchCodeResponse(message="entity_tools not available")
            
        try:
            self._activate_instance()
            # Search for classes and functions in this namespace
            all_entities = []
            for entity_type in ["class", "class_template", "function", "function_template"]:
                result = search_symbol(
                    type=entity_type,
                    namespace=namespace_name,
                    file=file_pattern,
                    limit=limit,
                )
                all_entities.extend(result.get("entities", []))
            
            # Use a dict for deduplication by ID
            unique_entities = {e.get("id") or e.get("fq_name"): e for e in all_entities if e.get("id") or e.get("fq_name")}
            
            hits = []
            for entity in list(unique_entities.values())[:limit]:
                hit = _entity_to_search_hit(entity, self._file_repo, skeleton=True)
                if hit:
                    hits.append(hit)
            
            message = f"Found {len(hits)} entity(ies) in namespace '{namespace_name}'"
            return SearchCodeResponse(message=message, hits=hits)
            
        except Exception as e:
            logger.error(f"Error in find_namespace: {e}", exc_info=True)
            return SearchCodeResponse(
                message=f"Error searching for namespace: {str(e)}"
            )

    def find_by_name(
        self,
        class_name: str = None,
        function_name: str = None,
        file_pattern: Optional[str] = None,
        **kwargs
    ) -> SearchCodeResponse:
        """
        Find by name (function or class).
        
        Args:
            class_name: Class name to search for
            function_name: Function name to search for
            file_pattern: Optional file pattern filter
        
        Returns:
            SearchCodeResponse with matching results
        """
        if function_name:
            return self.find_function(function_name, class_name, file_pattern)
        elif class_name:
            return self.find_class(class_name, file_pattern)
        else:
            return SearchCodeResponse(message="Either class_name or function_name must be provided")
