from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from neo4j import Driver, GraphDatabase

BATCH_SIZE = 256


def iter_jsonl(path: Path) -> Iterator[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunked(iterable: Iterable[Dict], size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Neo4jIndexer:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
    ) -> None:
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **parameters) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(query, **parameters)

    def init_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (feat:Feature) REQUIRE feat.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) REQUIRE t.id IS UNIQUE",
        ]
        for q in queries:
            self.run(q)

    def clear_graph(self) -> None:
        self.run("MATCH ()-[r]->() DELETE r")
        self.run("MATCH (n) DELETE n")

    def load_functions(self, path: Path, batch_size: int) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (f:Function {id: row.id})
        SET f += {
          fq_name: row.fq_name,
          short_name: row.short_name,
          def_file: row.def_file,
          def_start_line: row.def_start_line,
          def_end_line: row.def_end_line,
          namespaces: row.namespaces,
          classes: row.classes,
          is_method: row.is_method,
          is_static: row.is_static,
          is_template: row.is_template,
          return_type: row.return_type,
          parameter_types: row.parameter_types,
          is_test: coalesce(row.is_test, false)
        }
        """
        self._load_generic(path, query, batch_size)

    def load_classes(self, path: Path, batch_size: int) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (c:Class {id: row.id})
        SET c += {
          fq_name: row.fq_name,
          short_name: row.short_name,
          namespaces: row.namespaces,
          def_file: row.def_file,
          def_line: row.def_line,
          bases: row.bases,
          is_test: coalesce(row.is_test, false)
        }
        """
        self._load_generic(path, query, batch_size)

    def load_features(self, path: Path, batch_size: int) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (feat:Feature {id: row.feature_id})
        SET feat += {
          name: row.feature_name,
          description: row.feature_description,
          tags: row.tags,
          feature_type: row.feature_type,
          is_noise: coalesce(row.is_noise, false)
        }
        """
        self._load_generic(path, query, batch_size)
        self._relate_features(path, batch_size)

    def _relate_features(self, path: Path, batch_size: int) -> None:
        rel_q = """
        UNWIND $rows AS row
        MERGE (feat:Feature {id: row.feature_id})
        WITH feat, row
        UNWIND coalesce(row.member_funcs, []) AS fid
            MERGE (func:Function {id: fid})
            MERGE (feat)-[:COVERS {role: "member"}]->(func)
        """
        rep_q = """
        UNWIND $rows AS row
        MERGE (feat:Feature {id: row.feature_id})
        WITH feat, row
        UNWIND coalesce(row.representative_funcs, []) AS fid
            MERGE (func:Function {id: fid})
            MERGE (feat)-[:COVERS {role: "representative"}]->(func)
        """
        class_q = """
        UNWIND $rows AS row
        MERGE (feat:Feature {id: row.feature_id})
        WITH feat, row
        WHERE row.source_class IS NOT NULL
        MERGE (cls:Class {fq_name: row.source_class})
        MERGE (feat)-[:DERIVES_FROM]->(cls)
        """
        test_q = """
        UNWIND $rows AS row
        MERGE (feat:Feature {id: row.feature_id})
        WITH feat, row
        UNWIND coalesce(row.source_tests, []) AS tid
            MERGE (t:Test {id: tid})
            MERGE (feat)<-[:VALIDATES]-(t)
        """
        for q in (rel_q, rep_q, class_q, test_q):
            self._load_generic(path, q, batch_size)

    def load_callgraph(self, path: Path, batch_size: int) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (src:Function {id: row.caller})
        MERGE (dst:Function {id: row.callee})
        FOREACH (_ IN CASE WHEN row.edge_type = "call" THEN [1] ELSE [] END |
            MERGE (src)-[r:CALLS]->(dst)
            ON CREATE SET r.count = row.count
            ON MATCH SET r.count = coalesce(r.count, 0) + row.count
        )
        FOREACH (_ IN CASE WHEN row.edge_type <> "call" THEN [1] ELSE [] END |
            MERGE (src)-[r:DEPENDS_ON {edge_type: row.edge_type}]->(dst)
            ON CREATE SET r.count = row.count
            ON MATCH SET r.count = coalesce(r.count, 0) + row.count
        )
        """
        self._load_generic(path, query, batch_size)

    def _load_generic(self, path: Path, query: str, batch_size: int) -> None:
        if not path or not path.exists():
            return
        for batch in chunked(iter_jsonl(path), batch_size):
            self.run(query, rows=batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import JSONL data into Neo4j")
    parser.add_argument("--bolt-uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", help="Neo4j database name (optional)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--clear", action="store_true", help="Clear graph before import")
    parser.add_argument("--features", type=Path, default=Path("index_output/features.jsonl"))
    parser.add_argument("--functions", type=Path, default=Path("index_output/functions.jsonl"))
    parser.add_argument("--classes", type=Path, default=Path("index_output/classes.jsonl"))
    parser.add_argument("--callgraph", type=Path, default=Path("index_output/callgraph.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = Neo4jIndexer(args.bolt_uri, args.user, args.password, args.database)
    try:
        loader.init_constraints()
        if args.clear:
            loader.clear_graph()
        loader.load_functions(args.functions, args.batch_size)
        loader.load_classes(args.classes, args.batch_size)
        loader.load_features(args.features, args.batch_size)
        loader.load_callgraph(args.callgraph, args.batch_size)
    finally:
        loader.close()


if __name__ == "__main__":
    main()

