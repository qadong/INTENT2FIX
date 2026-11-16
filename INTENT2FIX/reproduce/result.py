
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from INTENT2FIX.node import Node
from INTENT2FIX.reproduce.intent_schema import ExecutionRecord, GraphBinding, TestArtifact

class ReproductionResult(BaseModel):
    """A structured object containing the results of a reproduction attempt."""
    
    is_reproduced: bool
    test_script: Optional[str] = None
    test_command: Optional[str] = None
    test_files: List[Dict[str, str]] = Field(default_factory=list)
    reproduction_output: Optional[str] = None
    expected_output: Optional[str] = None
    failure_class: Optional[str] = None
    matched_claim_ids: List[str] = Field(default_factory=list)
    test_artifacts: List[TestArtifact] = Field(default_factory=list)
    graph_bindings: List[GraphBinding] = Field(default_factory=list)
    execution_records: List[ExecutionRecord] = Field(default_factory=list)
    repository_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)
    full_trajectory: Optional[Node] = None
