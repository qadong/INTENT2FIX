from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from code_intent.code_intent.test_case_extractor import extract_test_cases

from INTENT2FIX.intent_dialogue.test_retrieval import _repository_test_paths
from INTENT2FIX.repository.repository import Repository


@dataclass(frozen=True)
class RepositoryTestEntry:
    path: str
    line: int
    framework: str
    name: str
    description: str = ""
    test_case_id: str = ""
    end_line: int = 0
    parent_name: str = ""
    body: str = ""
    oracle: str = ""

    @property
    def searchable_text(self) -> str:
        return "\n".join((self.name, self.description, self.path, self.framework))


def _preceding_comment(lines: list[str], line_number: int) -> str:
    index = line_number - 2
    while index >= 0 and not lines[index].strip():
        index -= 1
    comments: list[str] = []
    while index >= 0 and len(comments) < 3:
        text = lines[index].strip()
        if text.startswith("//"):
            comments.append(text[2:].strip())
        elif text.startswith("/*") or text.startswith("*") or text.endswith("*/"):
            comments.append(text.strip("/* "))
        else:
            break
        index -= 1
    return " ".join(reversed([item for item in comments if item]))[:500]


def _extract_file_tests(path: str, content: str) -> list[RepositoryTestEntry]:
    lines = content.splitlines()
    entries = []
    for case in extract_test_cases(path, content):
        line = int(case.get("def_start_line") or 1)
        name = str(case.get("fq_name") or "")
        description = _preceding_comment(lines, line)
        if not description:
            description = name.replace("_", " ")
        entries.append(RepositoryTestEntry(
            path=path,
            line=line,
            end_line=int(case.get("def_end_line") or line),
            framework=str(case.get("framework") or ""),
            name=name,
            description=description,
            test_case_id=str(case.get("id") or ""),
            parent_name=str(case.get("parent_name") or ""),
            body=str(case.get("code") or ""),
            oracle=str(case.get("oracle_code") or ""),
        ))
    return entries


class RepositoryTestInventory:
    """Lazily collect and regex-filter repository C/C++ test declarations."""

    def __init__(self, repository: Repository):
        self.repository = repository
        self._entries: list[RepositoryTestEntry] | None = None

    @property
    def entries(self) -> list[RepositoryTestEntry]:
        if self._entries is None:
            collected: list[RepositoryTestEntry] = []
            for path in _repository_test_paths(self.repository):
                content = self.repository.get_file_content(path) or ""
                if not content or len(content) > 750_000:
                    continue
                collected.extend(_extract_file_tests(path, content))
            self._entries = collected
        return self._entries

    def search(
        self,
        pattern: str,
        *,
        path_pattern: str = "",
        limit: int = 30,
    ) -> tuple[list[RepositoryTestEntry], int]:
        matcher = re.compile(pattern, re.IGNORECASE)
        matches = [
            entry
            for entry in self.entries
            if matcher.search(entry.searchable_text)
            and (
                not path_pattern
                or fnmatch.fnmatch(entry.path, path_pattern)
            )
        ]
        matches.sort(key=lambda entry: (
            0 if matcher.search(entry.name) else 1,
            0 if matcher.search(entry.description) else 1,
            entry.path,
            entry.line,
        ))
        return matches[:limit], len(matches)
