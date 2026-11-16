
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

INDEX_ROOT = Path("index")

ENTITY_TYPES = {
    "functions": "functions.jsonl",
    "classes": "classes.jsonl",
    "class_templates": "class_templates.jsonl",
    "function_templates": "function_templates.jsonl",
    "macros": "macros.jsonl",
    "template_instantiations": "template_instantiations.jsonl",
    "files": "files.jsonl",
}


class EntityQueryRequest(BaseModel):
    instance_name: str = Field(..., description="Instance name, e.g. swe-bench_catchorg__Catch2-2849")
    entity_type: str = Field(..., description=f"Entity type, options: {', '.join(ENTITY_TYPES.keys())}")
    name: Optional[str] = Field(None, description="Filter by name (partial match)")
    file: Optional[str] = Field(None, description="Filter by file path (partial match)")
    id: Optional[str] = Field(None, description="Exact match by ID")
    limit: int = Field(100, ge=1, le=1000, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class EntityQueryResponse(BaseModel):
    total: int = Field(..., description="Total count")
    count: int = Field(..., description="Current page count")
    offset: int = Field(..., description="Pagination offset")
    limit: int = Field(..., description="Pagination limit")
    entities: List[Dict[str, Any]] = Field(..., description="Entity list")


class InstanceListResponse(BaseModel):
    instances: List[str] = Field(..., description="Available instance names")


class EntityStatsResponse(BaseModel):
    instance_name: str
    stats: Dict[str, int] = Field(..., description="Count per entity type")


def _get_entities_dir(instance_name: str) -> Path:
    paths = [
        INDEX_ROOT / instance_name / "intent_output" / instance_name / "index_output" / "entities",
        INDEX_ROOT / instance_name / "index_output" / "entities",
    ]
    for path in paths:
        if path.exists():
            return path
    raise HTTPException(
        status_code=404,
        detail=f"Entity dir for instance '{instance_name}' not found. Tried: {[str(p) for p in paths]}"
    )


def _load_entities(
    entities_dir: Path,
    entity_type: str,
    name: Optional[str] = None,
    file: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if entity_type not in ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported entity type: {entity_type}. Supported: {', '.join(ENTITY_TYPES.keys())}"
        )
    
    file_path = entities_dir / ENTITY_TYPES[entity_type]
    if not file_path.exists():
        return []
    
    entities = []
    name_lower = name.lower() if name else None
    file_lower = file.lower() if file else None
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entity = json.loads(line)
                    
                    if entity_id and entity.get("id") != entity_id:
                        continue
                    
                    if name_lower:
                        display_name = entity.get("display_name", "").lower()
                        fq_name = entity.get("fq_name", "").lower()
                        short_name = entity.get("short_name", "").lower()
                        if name_lower not in display_name and name_lower not in fq_name and name_lower not in short_name:
                            continue
                    
                    if file_lower:
                        entity_file = entity.get("file", "").lower()
                        def_file = entity.get("def_file", "").lower()
                        trigger_file = entity.get("trigger_file", "").lower()
                        if (file_lower not in entity_file and 
                            file_lower not in def_file and 
                            file_lower not in trigger_file):
                            continue
                    
                    entities.append(entity)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read entity file: {str(e)}"
        )
    
    return entities


def _list_instances() -> List[str]:
    instances = []
    if not INDEX_ROOT.exists():
        return instances
    
    for item in INDEX_ROOT.iterdir():
        if item.is_dir():
            paths = [
                item / "intent_output" / item.name / "index_output" / "entities",
                item / "index_output" / "entities",
            ]
            for path in paths:
                if path.exists():
                    instances.append(item.name)
                    break
    
    return sorted(instances)


def _get_entity_stats(instance_name: str) -> Dict[str, int]:
    entities_dir = _get_entities_dir(instance_name)
    stats = {}
    
    for entity_type, filename in ENTITY_TYPES.items():
        file_path = entities_dir / filename
        if file_path.exists():
            count = 0
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            except Exception:
                pass
            stats[entity_type] = count
        else:
            stats[entity_type] = 0
    
    return stats


app = FastAPI(
    title="Code Intent Entity API",
    description="Query all extracted entities (functions, classes, macros, template instantiations, etc.)",
    version="1.0.0",
)


@app.get("/instances", response_model=InstanceListResponse)
def list_instances():
    instances = _list_instances()
    return InstanceListResponse(instances=instances)


@app.get("/instances/{instance_name}/stats", response_model=EntityStatsResponse)
def get_instance_stats(instance_name: str):
    try:
        stats = _get_entity_stats(instance_name)
        return EntityStatsResponse(instance_name=instance_name, stats=stats)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/entities/query", response_model=EntityQueryResponse)
def query_entities(request: EntityQueryRequest):
    try:
        entities_dir = _get_entities_dir(request.instance_name)
        all_entities = _load_entities(
            entities_dir=entities_dir,
            entity_type=request.entity_type,
            name=request.name,
            file=request.file,
            entity_id=request.id,
        )
        
        total = len(all_entities)
        start = request.offset
        end = request.offset + request.limit
        entities = all_entities[start:end]
        
        return EntityQueryResponse(
            total=total,
            count=len(entities),
            offset=request.offset,
            limit=request.limit,
            entities=entities,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/entities/query", response_model=EntityQueryResponse)
def query_entities_get(
    instance_name: str = Query(..., description="Instance name"),
    entity_type: str = Query(..., description=f"Entity type, options: {', '.join(ENTITY_TYPES.keys())}"),
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    file: Optional[str] = Query(None, description="Filter by file path (partial match)"),
    id: Optional[str] = Query(None, alias="entity_id", description="Exact match by ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    request = EntityQueryRequest(
        instance_name=instance_name,
        entity_type=entity_type,
        name=name,
        file=file,
        id=id,
        limit=limit,
        offset=offset,
    )
    return query_entities(request)


@app.get("/entities/{instance_name}/{entity_type}", response_model=EntityQueryResponse)
def get_entities_by_type(
    instance_name: str,
    entity_type: str,
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    file: Optional[str] = Query(None, description="Filter by file path (partial match)"),
    id: Optional[str] = Query(None, alias="entity_id", description="Exact match by ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    request = EntityQueryRequest(
        instance_name=instance_name,
        entity_type=entity_type,
        name=name,
        file=file,
        id=id,
        limit=limit,
        offset=offset,
    )
    return query_entities(request)


@app.get("/")
def root():
    return {
        "message": "Code Intent Entity API",
        "version": "1.0.0",
        "endpoints": {
            "GET /instances": "List available instances",
            "GET /instances/{instance_name}/stats": "Get entity stats for instance",
            "GET /entities/query": "Query entities (GET)",
            "POST /entities/query": "Query entities (POST)",
            "GET /entities/{instance_name}/{entity_type}": "Get entities by type",
        },
        "entity_types": list(ENTITY_TYPES.keys()),
    }

