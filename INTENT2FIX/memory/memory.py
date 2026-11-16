from pydantic import BaseModel, Field
from typing import Any, List, Dict

class Memory(BaseModel):
    instance_id: str
    problem_statement: str
    
    # New, specific summary fields
    reproduction_summary: str = ""
    repair_summary: str = ""

    # Patches
    solution_patch: str = ""
    golden_patch: str = ""

    # Reproduction data
    test_command: str = ""
    test_files: List[Dict[str, str]] = []  # To store {"path": "...", "content": "..."}
    reproduction_result: str = ""
    expected_reproduction_result: str = ""
    issue_evidence: Dict[str, Any] = Field(default_factory=dict)
    repair_intent_bundle: Dict[str, Any] = Field(default_factory=dict)
    repair_intent_contract: Dict[str, Any] = Field(default_factory=dict)
    reproduction_execution_records: List[Dict[str, Any]] = Field(default_factory=list)
