import re
import shlex
from pathlib import PurePosixPath


_NETWORK_EXECUTABLES = {
    "curl",
    "ftp",
    "gh",
    "lftp",
    "nc",
    "netcat",
    "scp",
    "sftp",
    "ssh",
    "wget",
}
_REMOTE_GIT_SUBCOMMANDS = {
    "clone",
    "fetch",
    "ls-remote",
    "pull",
}
_MUTATING_GIT_SUBCOMMANDS = {
    "add",
    "am",
    "apply",
    "checkout",
    "cherry-pick",
    "clean",
    "commit",
    "merge",
    "mv",
    "reset",
    "restore",
    "revert",
    "rm",
    "stash",
    "switch",
}
_FUTURE_DATA_REJECTION = (
    "curl/wget and remote Git commands are unavailable. Do not retrieve or "
    "reference future fixes; work only from the checked-out base repository."
)
_GIT_MUTATION_REJECTION = (
    "RunCommand cannot modify or revert repository files with Git. Use the "
    "provided file editing actions so every change is recorded in the current "
    "patch context."
)

_REPOSITORY_SCRIPT_SUFFIXES = {
    ".cmake",
    ".js",
    ".lua",
    ".mjs",
    ".php",
    ".pl",
    ".py",
    ".rb",
    ".sh",
    ".ts",
}
_SCRIPT_INTERPRETERS = {
    "bash",
    "node",
    "perl",
    "php",
    "python",
    "python2",
    "python3",
    "ruby",
    "sh",
}


def _unwrap_command_prefix(tokens: list[str]) -> list[str]:
    while tokens:
        executable = PurePosixPath(tokens[0]).name.lower()
        if executable == "command":
            tokens = [token for token in tokens[1:] if not token.startswith("-")]
            continue
        if executable == "env":
            index = 1
            while index < len(tokens) and (
                tokens[index].startswith("-") or "=" in tokens[index]
            ):
                index += 1
            tokens = tokens[index:]
            continue
        if executable == "timeout":
            index = 1
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            if index < len(tokens):
                index += 1
            tokens = tokens[index:]
            continue
        break
    return tokens


def _tokens_may_execute_repository_code(tokens: list[str]) -> bool:
    tokens = _unwrap_command_prefix(tokens)
    if not tokens:
        return False

    executable_token = tokens[0]
    executable = PurePosixPath(executable_token).name.lower()
    if executable in _SCRIPT_INTERPRETERS:
        return any(
            token == "-m"
            or (
                not token.startswith("-")
                and (
                    "/" in token
                    or PurePosixPath(token).suffix.lower()
                    in _REPOSITORY_SCRIPT_SUFFIXES
                )
            )
            for token in tokens[1:]
        )
    if executable in {"make", "gmake", "ninja"}:
        return True
    if executable == "cmake" and any(token in {"--build", "-P"} for token in tokens[1:]):
        return True
    if executable in {"npm", "pnpm", "yarn"} and "run" in tokens[1:]:
        return True
    if executable == "cargo" and "xtask" in tokens[1:]:
        return True

    candidate = PurePosixPath(executable_token)
    return (
        executable_token.startswith("./")
        or (not candidate.is_absolute() and "/" in executable_token)
        or candidate.suffix.lower() in _REPOSITORY_SCRIPT_SUFFIXES
    )


def may_execute_repository_code(command: str) -> bool:
    """Return whether a command can execute repository-owned code or a target.

    This is intentionally based on command shape rather than generator names.
    Actual write detection decides whether the command behaved as a generator.
    """
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if _tokens_may_execute_repository_code(tokens):
            return True
    return False


def future_data_command_rejection(command: str) -> str:
    """Reject common shell commands that can retrieve benchmark future data."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    executables = {token.rsplit("/", 1)[-1] for token in tokens}
    if executables & _NETWORK_EXECUTABLES:
        return _FUTURE_DATA_REJECTION

    for index, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] != "git":
            continue
        if any(candidate in _REMOTE_GIT_SUBCOMMANDS for candidate in tokens[index + 1 :]):
            return _FUTURE_DATA_REJECTION

    if any(path in command for path in ("/home/fix.patch", "/home/test.patch")):
        return (
            "Benchmark fix and test patches are unavailable. Do not inspect or apply "
            "reference patches; derive the repair from the issue and base repository."
        )
    return ""


def local_git_mutation_rejection(command: str) -> str:
    """Reject Git commands whose state changes bypass the patch context."""
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        tokens = _unwrap_command_prefix(tokens)
        if not tokens or PurePosixPath(tokens[0]).name.lower() != "git":
            continue

        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            option = tokens[index].split("=", 1)[0]
            index += 1
            if option in {
                "-C",
                "-c",
                "--config-env",
                "--git-dir",
                "--namespace",
                "--super-prefix",
                "--work-tree",
            } and "=" not in tokens[index - 1]:
                index += 1
        if (
            index < len(tokens)
            and tokens[index].lower() in _MUTATING_GIT_SUBCOMMANDS
        ):
            return _GIT_MUTATION_REJECTION
    return ""


def verification_exit_status_rejection(command: str) -> str:
    """Reject shell composition that can hide a focused test's exit status."""
    lexer = shlex.shlex(
        command,
        posix=True,
        punctuation_chars="|;",
    )
    lexer.whitespace_split = True
    try:
        operators = {token for token in lexer if token in {"|", "||", ";"}}
    except ValueError:
        return (
            "Focused verification command could not be parsed. Run the test "
            "directly with a simple command whose exit status is authoritative."
        )
    if operators:
        return (
            "Focused verification must preserve the test program's real exit "
            "status. Do not use shell pipelines, `||`, or `;` here because a "
            "later grep/head/echo command can turn a failure into exit code 0. "
            "Run the compile/test command directly (chaining prerequisites with "
            "`&&` is allowed), then inspect extra output with RunCommand."
        )
    return ""
