import os
import json
import subprocess
import py_compile
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 2
REPOSITORY_LIMIT = 30000
IMPLEMENTATION_CONTEXT_LIMIT = 12000


class AgentState(TypedDict, total=False):
    task: str
    plan: list[str]
    skills: list[str]

    repository: str
    search_result: str

    implementation: str
    changed_files: list[str]

    test_result: str
    test_passed: bool

    review: str
    documentation: str

    retry_count: int
    final_result: str


SKILLS = {
    "requirement_analysis",
    "codebase_understanding",
    "architecture",
    "implementation",
    "debugging",
    "testing",
    "code_review",
    "documentation",
    "git",
    "dependency_analysis",
    "ci_cd",
    "security"
}


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build"
}


SOURCE_EXTENSIONS = (
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".md"
)


def list_files():
    files = []

    for root, dirs, filenames in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRS
        ]

        for filename in filenames:
            path = os.path.join(
                root,
                filename
            )

            if path.endswith(SOURCE_EXTENSIONS):
                files.append(path)

    files.sort(
        key=lambda path: (
            0 if path.endswith(".py") else 1,
            path.lower()
        )
    )

    return files


def read_file(path):
    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return file.read()
    except Exception:
        return ""


def read_repository(limit=REPOSITORY_LIMIT):
    files = list_files()

    output = []
    total = 0

    for path in files:
        content = read_file(path)

        if not content.strip():
            continue

        remaining = limit - total

        if remaining <= 0:
            break

        content = content[:remaining]

        output.append(
            f"\n--- FILE: {path} ---\n"
            f"{content}\n"
        )

        total += len(content)

    if not output:
        return "No source files found."

    return "\n".join(output)


def search_repository(query):
    query = query.strip()

    if not query:
        return "Search query was empty."

    results = []

    for path in list_files():
        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:
                lines = file.readlines()

        except Exception:
            continue

        for line_number, line in enumerate(
            lines,
            start=1
        ):
            if query.lower() in line.lower():
                results.append(
                    f"{path}:{line_number}: "
                    f"{line.strip()}"
                )

                if len(results) >= 100:
                    return (
                        f'Found "{query}" in '
                        f"{len(results)}+ locations:\n\n"
                        + "\n".join(results)
                    )

    if not results:
        return (
            f'No occurrence of "{query}" '
            "was found in the project."
        )

    return (
        f'Found "{query}" in '
        f"{len(results)} location(s):\n\n"
        + "\n".join(results)
    )


def safe_path(path):
    if not isinstance(path, str):
        return False

    path = path.strip()

    if not path:
        return False

    normalized = os.path.normpath(path)

    if os.path.isabs(normalized):
        return False

    if normalized == ".":
        return False

    if normalized == "..":
        return False

    if normalized.startswith(".." + os.sep):
        return False

    return True


def write_file(path, content):
    if not safe_path(path):
        raise ValueError(
            f"Unsafe path rejected: {path}"
        )

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        file.write(content)


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        ).strip()

        return {
            "code": result.returncode,
            "output": output
        }

    except subprocess.TimeoutExpired:
        return {
            "code": -1,
            "output": "Command timed out."
        }

    except Exception as e:
        return {
            "code": -1,
            "output": str(e)
        }


def get_json(text):
    if not isinstance(text, str):
        raise ValueError(
            "Model response is not text."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found."
            )

        return json.loads(
            text[start:end + 1]
        )


def call_llm(llm, prompt):
    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    return response.content


def planner(state, llm):
    prompt = f"""
You are the planner of a Software Engineering Agent.

USER TASK:
{state["task"]}

AVAILABLE SKILLS:

requirement_analysis
codebase_understanding
architecture
implementation
debugging
testing
code_review
documentation
git
dependency_analysis
ci_cd
security

PRIMARY SKILLS:

implementation
code_review
documentation

Select only the skills actually required.

Important:

- If the user asks to create or modify code, use implementation.
- If the user asks to find/search/locate code, use codebase_understanding.
- If the user asks to review code, use code_review.
- If the user asks for documentation, use documentation.
- If the user asks to fix an error, use debugging.
- If validation is required, use testing.

Return ONLY JSON.

The response must contain the word JSON.

Format:

{{
    "plan": [
        "step 1",
        "step 2"
    ],
    "skills": [
        "implementation"
    ]
}}
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        data = get_json(content)

        plan = data.get(
            "plan",
            []
        )

        skills = data.get(
            "skills",
            []
        )

        if not isinstance(plan, list):
            plan = []

        if not isinstance(skills, list):
            skills = []

        skills = [
            skill
            for skill in skills
            if skill in SKILLS
        ]

        task = state["task"].lower()

        coding_words = (
            "create",
            "build",
            "implement",
            "write",
            "develop",
            "modify",
            "add",
            "code",
            "function",
            "program"
        )

        if any(
            word in task
            for word in coding_words
        ):
            if "implementation" not in skills:
                skills.append(
                    "implementation"
                )

        if not skills:
            skills = [
                "requirement_analysis"
            ]

        return {
            "plan": plan,
            "skills": skills
        }

    except Exception as e:
        return {
            "plan": [
                "Understand the requirement",
                "Implement the requested solution"
            ],
            "skills": [
                "requirement_analysis",
                "implementation"
            ]
        }


def requirement_node(state):
    return {
        "plan": state.get(
            "plan",
            []
        )
    }


def repository_node(state):
    return {
        "repository": read_repository()
    }


def search_node(state, llm):
    prompt = f"""
You are a codebase search assistant.

USER TASK:
{state["task"]}

Return the most useful literal search term.

Return ONLY JSON.

The response must contain the word JSON.

Format:

{{
    "query": "term"
}}
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        data = get_json(content)

        query = data.get(
            "query",
            ""
        )

        if not query:
            query = state["task"]

        return {
            "search_result": search_repository(
                query
            )
        }

    except Exception as e:
        return {
            "search_result": (
                f"Search failed: {e}"
            )
        }


def codebase_understanding_node(state, llm):
    repository = state.get(
        "repository",
        ""
    )

    repository = repository[
        :IMPLEMENTATION_CONTEXT_LIMIT
    ]

    prompt = f"""
You are a software engineer analyzing an existing codebase.

USER TASK:
{state["task"]}

REPOSITORY:

{repository}

Explain only the relevant parts.

Identify:

- Relevant files
- Relevant functions/classes
- How the requested functionality currently works
- What needs to change, if anything

Do not modify files.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "search_result": content.strip()
        }

    except Exception as e:
        return {
            "search_result": (
                f"Codebase analysis failed: {e}"
            )
        }


def implementation_node(state, llm):
    task = state["task"]

    repository = state.get(
        "repository",
        ""
    )

    skills = state.get(
        "skills",
        []
    )

    needs_existing_code = (
        "codebase_understanding" in skills
        or "modify" in task.lower()
        or "change" in task.lower()
        or "fix" in task.lower()
        or "existing" in task.lower()
    )

    if needs_existing_code:
        repository = repository[
            :IMPLEMENTATION_CONTEXT_LIMIT
        ]
    else:
        repository = (
            "No existing repository context is required "
            "for this new/simple implementation."
        )

    prompt = f"""
You are the Implementation component of a Software Engineering Agent.

USER TASK:
{task}

EXISTING REPOSITORY CONTEXT:
{repository}

Implement the requested functionality.

Rules:

1. Write real executable code.
2. If Python is requested, write valid Python.
3. Create a new file when necessary.
4. Modify existing files only when necessary.
5. Return COMPLETE content for every changed file.
6. Never return partial file content.
7. Preserve existing functionality.
8. Use relative paths only.
9. Never use ../ paths.
10. Never use absolute paths.
11. Do not include markdown code fences inside file content.
12. Return ONLY JSON.
13. The response must contain the word JSON.

Format:

{{
    "summary": "what was implemented",
    "files": [
        {{
            "path": "example.py",
            "content": "complete file content"
        }}
    ]
}}
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        data = get_json(content)

        files = data.get(
            "files",
            []
        )

        changed_files = []

        for item in files:
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            file_content = item.get(
                "content"
            )

            if not safe_path(path):
                continue

            if not isinstance(
                file_content,
                str
            ):
                continue

            write_file(
                path,
                file_content
            )

            changed_files.append(
                path
            )

        if not changed_files:
            raise ValueError(
                "Model returned no valid files."
            )

        return {
            "implementation": data.get(
                "summary",
                "Implementation completed."
            ),
            "changed_files": changed_files
        }

    except Exception as e:
        return {
            "implementation": (
                f"Implementation failed: {e}"
            ),
            "changed_files": []
        }


def validate_python_files(changed_files):
    python_files = [
        path
        for path in changed_files
        if path.endswith(".py")
        and os.path.exists(path)
    ]

    if not python_files:
        return {
            "passed": True,
            "output": (
                "No changed Python files "
                "required syntax validation."
            )
        }

    errors = []

    for path in python_files:
        try:
            py_compile.compile(
                path,
                doraise=True
            )

        except Exception as e:
            errors.append(
                f"{path}: {e}"
            )

    if errors:
        return {
            "passed": False,
            "output": (
                "Python validation failed:\n"
                + "\n".join(errors)
            )
        }

    return {
        "passed": True,
        "output": (
            "Python syntax validation passed."
        )
    }


def has_pytest_tests():
    if os.path.isdir("tests"):
        for root, dirs, files in os.walk("tests"):
            for file in files:
                if (
                    file.startswith("test_")
                    and file.endswith(".py")
                ):
                    return True

    for path in list_files():
        filename = os.path.basename(path)

        if filename.startswith("test_"):
            return True

        if filename.endswith("_test.py"):
            return True

    return False


def testing_node(state):
    validation = validate_python_files(
        state.get(
            "changed_files",
            []
        )
    )

    if not validation["passed"]:
        return {
            "test_result": validation["output"],
            "test_passed": False
        }

    output = validation["output"]

    if has_pytest_tests():
        result = run_command(
            "python -m pytest -q"
        )

        output += (
            "\n\nPytest:\n"
            + result["output"]
        )

        return {
            "test_result": output,
            "test_passed": (
                result["code"] == 0
            )
        }

    output += (
        "\n\nNo pytest test suite detected."
    )

    return {
        "test_result": output,
        "test_passed": True
    }


def get_changed_code(state):
    output = []

    for path in state.get(
        "changed_files",
        []
    ):
        if not os.path.exists(path):
            continue

        content = read_file(path)

        output.append(
            f"\n--- FILE: {path} ---\n"
            f"{content}"
        )

    return "\n".join(output)


def debugging_node(state, llm):
    code = get_changed_code(
        state
    )

    code = code[
        :IMPLEMENTATION_CONTEXT_LIMIT
    ]

    prompt = f"""
You are a debugging engineer.

TASK:
{state["task"]}

CURRENT CODE:
{code}

VALIDATION ERROR:
{state.get("test_result", "")}

Fix the problem.

Return ONLY JSON.

The response must contain the word JSON.

Format:

{{
    "files": [
        {{
            "path": "example.py",
            "content": "complete corrected file"
        }}
    ]
}}
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        data = get_json(content)

        changed_files = list(
            state.get(
                "changed_files",
                []
            )
        )

        for item in data.get(
            "files",
            []
        ):
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            file_content = item.get(
                "content"
            )

            if not safe_path(path):
                continue

            if not isinstance(
                file_content,
                str
            ):
                continue

            write_file(
                path,
                file_content
            )

            if path not in changed_files:
                changed_files.append(
                    path
                )

        return {
            "retry_count": (
                state.get(
                    "retry_count",
                    0
                ) + 1
            ),
            "changed_files": changed_files
        }

    except Exception:
        return {
            "retry_count": (
                state.get(
                    "retry_count",
                    0
                ) + 1
            )
        }


def review_node(state, llm):
    code = get_changed_code(
        state
    )

    code = code[
        :IMPLEMENTATION_CONTEXT_LIMIT
    ]

    prompt = f"""
You are a senior software engineer performing a code review.

TASK:
{state["task"]}

CODE:
{code}

VALIDATION:
{state.get("test_result", "")}

Review the implementation for:

- Correctness
- Logic
- Readability
- Maintainability
- Security
- Unnecessary code
- Requirement satisfaction

Return a concise normal-text review.

Do not modify files.
Do not return JSON.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "review": content.strip()
        }

    except Exception as e:
        return {
            "review": (
                f"Review failed: {e}"
            )
        }


def documentation_node(state, llm):
    prompt = f"""
You are a technical documentation engineer.

TASK:
{state["task"]}

IMPLEMENTATION:
{state.get("implementation", "")}

CHANGED FILES:
{state.get("changed_files", [])}

VALIDATION:
{state.get("test_result", "")}

CODE REVIEW:
{state.get("review", "")}

Write concise technical documentation.

Include:

1. What changed
2. Files changed
3. How it works
4. How to use it

Maximum 150 words.

Return normal text only.
Do not return JSON.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "documentation": content.strip()
        }

    except Exception as e:
        return {
            "documentation": (
                f"Documentation failed: {e}"
            )
        }


def architecture_node(state, llm):
    repository = state.get(
        "repository",
        ""
    )[:IMPLEMENTATION_CONTEXT_LIMIT]

    prompt = f"""
You are a software architect.

TASK:
{state["task"]}

REPOSITORY:
{repository}

Explain the relevant architecture.

Cover:

- Main components
- Their responsibilities
- How they interact
- Important design decisions
- Possible improvements

Return normal text only.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "review": content.strip()
        }

    except Exception as e:
        return {
            "review": (
                f"Architecture analysis failed: {e}"
            )
        }


def dependency_node(state, llm):
    repository = state.get(
        "repository",
        ""
    )[:IMPLEMENTATION_CONTEXT_LIMIT]

    prompt = f"""
You are a dependency analysis engineer.

TASK:
{state["task"]}

REPOSITORY:
{repository}

Identify relevant:

- Libraries
- Frameworks
- Dependencies
- Version information
- Missing dependencies
- Potential dependency problems

Return normal text only.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "review": content.strip()
        }

    except Exception as e:
        return {
            "review": (
                f"Dependency analysis failed: {e}"
            )
        }


def security_node(state, llm):
    repository = state.get(
        "repository",
        ""
    )[:IMPLEMENTATION_CONTEXT_LIMIT]

    prompt = f"""
You are a software security reviewer.

TASK:
{state["task"]}

REPOSITORY:
{repository}

Look for:

- Hardcoded secrets
- Unsafe file operations
- Command execution
- Input validation problems
- Sensitive information exposure
- Authentication problems
- Dangerous dependencies

Return concise normal-text findings.

Do not modify files.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "review": content.strip()
        }

    except Exception as e:
        return {
            "review": (
                f"Security analysis failed: {e}"
            )
        }


def git_node(state):
    result = run_command(
        "git status --short"
    )

    return {
        "review": (
            "Git status:\n"
            + result["output"]
        )
    }


def ci_cd_node(state, llm):
    repository = state.get(
        "repository",
        ""
    )[:IMPLEMENTATION_CONTEXT_LIMIT]

    prompt = f"""
You are a CI/CD troubleshooting engineer.

TASK:
{state["task"]}

REPOSITORY:
{repository}

Analyze relevant CI/CD configuration.

Look for:

- GitHub Actions
- GitLab CI
- Jenkins
- Docker
- Build configuration
- Test configuration
- Deployment configuration

Explain likely problems and solutions.

Return normal text only.
"""

    try:
        content = call_llm(
            llm,
            prompt
        )

        return {
            "review": content.strip()
        }

    except Exception as e:
        return {
            "review": (
                f"CI/CD analysis failed: {e}"
            )
        }


def final_node(state):
    if state.get("test_passed") is False:
        status = "Completed with validation issues."
    else:
        status = "Completed."

    files = ", ".join(
        state.get(
            "changed_files",
            []
        )
    )

    if not files:
        files = "No files changed."

    result = f"""
FINAL RESULT

Status:
{status}

Plan:
{state.get("plan", [])}

Selected skills:
{state.get("skills", [])}

Files changed:
{files}

Search / Analysis:
{state.get("search_result", "Not performed.")}

Implementation:
{state.get("implementation", "Not performed.")}

Testing:
{state.get("test_result", "Not performed.")}

Code Review:
{state.get("review", "Not performed.")}

Documentation:
{state.get("documentation", "Not performed.")}
"""

    return {
        "final_result": result.strip()
    }


def choose_after_planner(state):
    skills = state.get(
        "skills",
        []
    )

    if (
        "implementation" in skills
        or "codebase_understanding" in skills
        or "architecture" in skills
        or "dependency_analysis" in skills
        or "security" in skills
        or "ci_cd" in skills
        or "code_review" in skills
        or "documentation" in skills
    ):
        return "repository"

    if "git" in skills:
        return "git"

    return "final"


def choose_after_repository(state):
    skills = state.get(
        "skills",
        []
    )

    task = state["task"].lower()

    search_words = (
        "find",
        "search",
        "where",
        "locate",
        "similar",
        "used",
        "usage",
        "contains"
    )

    if (
        "codebase_understanding" in skills
        and any(
            word in task
            for word in search_words
        )
        and "implementation" not in skills
    ):
        return "understanding"

    if "architecture" in skills:
        return "architecture"

    if "dependency_analysis" in skills:
        return "dependency"

    if "security" in skills:
        return "security"

    if "ci_cd" in skills:
        return "ci_cd"

    if "implementation" in skills:
        return "implementation"

    if "code_review" in skills:
        return "review"

    if "documentation" in skills:
        return "documentation"

    return "final"


def choose_after_analysis(state):
    skills = state.get(
        "skills",
        []
    )

    if "implementation" in skills:
        return "implementation"

    if "code_review" in skills:
        return "review"

    if "documentation" in skills:
        return "documentation"

    return "final"


def choose_after_testing(state):
    if not state.get(
        "test_passed",
        False
    ):
        if state.get(
            "retry_count",
            0
        ) < MAX_RETRIES:
            return "debug"

    skills = state.get(
        "skills",
        []
    )

    if "code_review" in skills:
        return "review"

    if "documentation" in skills:
        return "documentation"

    return "final"


def choose_after_review(state):
    if "documentation" in state.get(
        "skills",
        []
    ):
        return "documentation"

    return "final"


def build_graph(llm):
    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "planner",
        lambda state:
        planner(
            state,
            llm
        )
    )

    graph.add_node(
        "requirement",
        requirement_node
    )

    graph.add_node(
        "repository",
        repository_node
    )

    graph.add_node(
        "understanding",
        lambda state:
        codebase_understanding_node(
            state,
            llm
        )
    )

    graph.add_node(
        "implementation",
        lambda state:
        implementation_node(
            state,
            llm
        )
    )

    graph.add_node(
        "testing",
        testing_node
    )

    graph.add_node(
        "debug",
        lambda state:
        debugging_node(
            state,
            llm
        )
    )

    graph.add_node(
        "review",
        lambda state:
        review_node(
            state,
            llm
        )
    )

    graph.add_node(
        "documentation",
        lambda state:
        documentation_node(
            state,
            llm
        )
    )

    graph.add_node(
        "architecture",
        lambda state:
        architecture_node(
            state,
            llm
        )
    )

    graph.add_node(
        "dependency",
        lambda state:
        dependency_node(
            state,
            llm
        )
    )

    graph.add_node(
        "security",
        lambda state:
        security_node(
            state,
            llm
        )
    )

    graph.add_node(
        "git",
        git_node
    )

    graph.add_node(
        "ci_cd",
        lambda state:
        ci_cd_node(
            state,
            llm
        )
    )

    graph.add_node(
        "final",
        final_node
    )

    graph.add_edge(
        START,
        "planner"
    )

    graph.add_edge(
        "planner",
        "requirement"
    )

    graph.add_conditional_edges(
        "requirement",
        choose_after_planner,
        {
            "repository": "repository",
            "git": "git",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "repository",
        choose_after_repository,
        {
            "understanding": "understanding",
            "architecture": "architecture",
            "dependency": "dependency",
            "security": "security",
            "ci_cd": "ci_cd",
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "understanding",
        choose_after_analysis,
        {
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "architecture",
        choose_after_analysis,
        {
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "dependency",
        choose_after_analysis,
        {
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "security",
        choose_after_analysis,
        {
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_conditional_edges(
        "ci_cd",
        choose_after_analysis,
        {
            "implementation": "implementation",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_edge(
        "git",
        "final"
    )

    graph.add_edge(
        "implementation",
        "testing"
    )

    graph.add_conditional_edges(
        "testing",
        choose_after_testing,
        {
            "debug": "debug",
            "review": "review",
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_edge(
        "debug",
        "testing"
    )

    graph.add_conditional_edges(
        "review",
        choose_after_review,
        {
            "documentation": "documentation",
            "final": "final"
        }
    )

    graph.add_edge(
        "documentation",
        "final"
    )

    graph.add_edge(
        "final",
        END
    )

    return graph.compile()


def main():
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0
    )

    app = build_graph(
        llm
    )

    task = input(
        "\nEnter software engineering task:\n> "
    ).strip()

    if not task:
        print(
            "\nPlease enter a software engineering task."
        )
        return

    print(
        "\nAgent started..."
    )

    try:
        result = app.invoke(
            {
                "task": task,
                "retry_count": 0
            }
        )

        print(
            "\n"
            + result.get(
                "final_result",
                "Agent did not produce a final result."
            )
        )

    except Exception as e:
        print(
            "\nAgent execution failed:"
        )
        print(e)


if __name__ == "__main__":
    main()