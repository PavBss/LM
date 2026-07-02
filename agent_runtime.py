import json
import os
import re
import shlex
import subprocess
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from main import GenerationDisplay, effective_generation_tokens, generate_streaming, load_model_for_inference, render_chat_prompt, unwrap_model


ROOT_DIR = Path(__file__).resolve().parent
ROOT_DIR_RESOLVED = ROOT_DIR.resolve()
TOOL_PATTERN = re.compile(r"<TOOL_CALL>(.*?)</TOOL_CALL>", re.DOTALL)
ALLOWED_SHELL_COMMANDS = {"python", "python3", "py", "pip", "pip3", "pytest", "ls", "dir", "cat", "type", "sed", "rg"}
ALLOWED_GIT_SUBCOMMANDS = {"status", "diff"}
SHELL_CONTROL_TOKENS = ("&&", "||", ";", "|", "`", "$(", ">", ">>", "<")
DESTRUCTIVE_PATTERNS = [
    re.compile(r"(?i)\brm\s+-[^;\n]*r[^;\n]*f\b"),
    re.compile(r"(?i)\bdel\s+/.+"),
    re.compile(r"(?i)\bformat\b"),
    re.compile(r"(?i)\bshutdown\b"),
    re.compile(r"(?i)\breboot\b"),
    re.compile(r"(?i)\bgit\s+reset\s+--hard\b"),
    re.compile(r"(?i)\bgit\s+checkout\s+--\b"),
]


AGENT_SYSTEM_PROMPT = """You are Clyx Coding Agent. You always plan before you act.

WORKFLOW:
1. PLAN FIRST. For a new task, output a plan: <PLAN>1. step\n2. step\n...</PLAN>. Keep it short and concrete.
2. CLARIFY. If the task is unclear or you miss information, do NOT guess. Output a clarifying question instead:
   <ASK>your question here?</ASK><OPTIONS><OPT>first option</OPT><OPT>second option</OPT><OPT>third option</OPT></OPTIONS>
   Always give exactly 3 options. The runtime can add a 4th custom answer. After the user answers, continue.
3. APPROVAL. After <PLAN>, wait. The runtime shows the plan to the user and asks to start.
4. EXECUTE. Only after approval, save the plan to plan.md, then run tools:
   <think>...</think><MODEL><TOOL_CALL>{...}</TOOL_CALL></MODEL> and wait for the tool result.

RULES:
- Output a short <think>...</think> block before each tool call or final answer (under 64 tokens).
- Never call a tool before the plan is approved.
- Keep the final answer under 128 tokens.

Supported tools:
- run_shell: {"name":"run_shell","arguments":{"cmd":"python --version"}}
- read_file: {"name":"read_file","arguments":{"path":"relative/file.py"}}
- write_file: {"name":"write_file","arguments":{"path":"relative/file.py","content":"text"}}
- list_dir: {"name":"list_dir","arguments":{"path":"."}}

After tool results, output <think>...</think><MODEL>final concise explanation</MODEL><|endtext|>.
"""


def confirm_tool(tool_call: Dict[str, Any]) -> bool:
    print("\n[Tool request]")
    print(json.dumps(tool_call, ensure_ascii=False, indent=2))
    answer = input("Allow this tool call? Type yes to run: ").strip().lower()
    return answer in {"yes", "y", "да", "д"}


def resolve_root(root: Optional[str] = None) -> Path:
    return Path(root).expanduser().resolve() if root else ROOT_DIR_RESOLVED


def safe_path(raw_path: str, root: Optional[str] = None) -> Path:
    root_dir = resolve_root(root)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {raw_path}") from exc
    return resolved


def validate_shell_command(cmd: str) -> Tuple[bool, str]:
    stripped = cmd.strip()
    if not stripped:
        return False, "missing cmd"
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(stripped):
            return False, "destructive command pattern is blocked"
    if any(token in stripped for token in SHELL_CONTROL_TOKENS):
        return False, "shell chaining, pipes and redirection are blocked"
    try:
        parts = shlex.split(stripped, posix=os.name != "nt")
    except ValueError as exc:
        return False, f"could not parse command: {exc}"
    if not parts:
        return False, "empty command"
    command = Path(parts[0]).name.lower()
    if command == "git":
        if len(parts) >= 2 and parts[1].lower() in ALLOWED_GIT_SUBCOMMANDS:
            return True, "allowed"
        return False, "only git status and git diff are allowed"
    if command not in ALLOWED_SHELL_COMMANDS:
        return False, f"command is not allowlisted: {command}"
    return True, "allowed"


def log_tool_event(log_path: Optional[Path], event: Dict[str, Any]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **event}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_tool(
    tool_call: Dict[str, Any],
    root: Optional[str] = None,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> str:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return "Tool error: arguments must be an object."

    root_dir = resolve_root(root)
    result_text = ""
    try:
        if name == "run_shell":
            cmd = str(args.get("cmd", ""))
            ok, reason = validate_shell_command(cmd)
            if not ok:
                result_text = f"Tool denied: {reason}."
                return result_text
            if dry_run:
                result_text = f"dry_run: shell command allowed but not executed: {cmd}"
                return result_text
            timeout = max(1, min(int(args.get("timeout", 120)), 300))
            result = subprocess.run(
                cmd,
                cwd=str(root_dir),
                shell=True,
                text=True,
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            result_text = f"exit_code={result.returncode}\n{output[-12000:]}"
            return result_text

        if name == "read_file":
            path = safe_path(str(args.get("path", "")), root=str(root_dir))
            if not path.exists() or not path.is_file():
                result_text = f"Tool error: file not found: {path}"
                return result_text
            if dry_run:
                result_text = f"dry_run: would read {path}"
                return result_text
            result_text = path.read_text(encoding="utf-8", errors="ignore")[:20000]
            return result_text

        if name == "write_file":
            path = safe_path(str(args.get("path", "")), root=str(root_dir))
            content = str(args.get("content", ""))
            if dry_run:
                result_text = f"dry_run: would write {len(content)} characters to {path}"
                return result_text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            result_text = f"wrote {len(content)} characters to {path}"
            return result_text

        if name == "list_dir":
            path = safe_path(str(args.get("path", ".")), root=str(root_dir))
            if not path.exists() or not path.is_dir():
                result_text = f"Tool error: directory not found: {path}"
                return result_text
            if dry_run:
                result_text = f"dry_run: would list {path}"
                return result_text
            rows = []
            for item in sorted(path.iterdir())[:200]:
                kind = "dir " if item.is_dir() else "file"
                rows.append(f"{kind} {item.name}")
            result_text = "\n".join(rows)
            return result_text

        result_text = f"Tool error: unknown tool: {name}"
        return result_text
    except Exception as exc:
        result_text = f"Tool exception: {type(exc).__name__}: {exc}"
        return result_text
    finally:
        log_tool_event(
            log_path,
            {
                "tool_call": tool_call,
                "root": str(root_dir),
                "dry_run": bool(dry_run),
                "result_preview": result_text[:2000],
            },
        )


def extract_tool_call(text: str):
    match = TOOL_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"name": "invalid_json", "arguments": {"error": str(exc), "raw": raw}}


PLAN_PATTERN = re.compile(r"<PLAN>(.*?)</PLAN>", re.DOTALL)
ASK_PATTERN = re.compile(r"<ASK>(.*?)</ASK>", re.DOTALL)
OPTIONS_PATTERN = re.compile(r"<OPTIONS>(.*?)</OPTIONS>", re.DOTALL)
OPT_PATTERN = re.compile(r"<OPT>(.*?)</OPT>", re.DOTALL)


def extract_plan(text: str) -> Optional[str]:
    """Return the plan body from <PLAN>...</PLAN>, or None when the model did not emit a plan."""
    match = PLAN_PATTERN.search(text)
    if not match:
        return None
    plan_body = match.group(1).strip()
    return plan_body or None


def extract_ask(text: str) -> Optional[Dict[str, Any]]:
    """Return {question, options} when the model asks a clarifying question via <ASK>...</ASK>."""
    ask_match = ASK_PATTERN.search(text)
    if not ask_match:
        return None
    question = ask_match.group(1).strip()
    options: List[str] = []
    options_match = OPTIONS_PATTERN.search(text)
    if options_match:
        options = [opt.strip() for opt in OPT_PATTERN.findall(options_match.group(1)) if opt.strip()]
    return {"question": question, "options": options}


def save_plan_to_file(plan_body: str, root: Optional[str] = None, dry_run: bool = False, log_path: Optional[Path] = None) -> str:
    """Write the approved plan to plan.md using the write_file tool, so it follows the same safety/logging path."""
    tool_call = {"name": "write_file", "arguments": {"path": "plan.md", "content": plan_body + "\n"}}
    return run_tool(tool_call, root=root, dry_run=dry_run, log_path=log_path)


def format_plan_for_display(plan_body: str) -> str:
    """Render a plan body as a clean numbered outline for the user."""
    plan_body = plan_body.strip()
    return plan_body


def prompt_for_answer(ask: Dict[str, Any]) -> str:
    """Show the model's clarifying question and its options; return the user's chosen answer text.

    Options 1..3 come from the model's <OPTIONS>; option 4 lets the user type a custom answer.
    """
    question = ask.get("question", "")
    options = [opt for opt in ask.get("options", []) if opt]
    print("\n[Clarifying question]")
    print(question)
    shown = options[:3]
    for idx, opt in enumerate(shown, 1):
        print(f"  {idx}) {opt}")
    print(f"  4) enter your ")
    while True:
        try:
            raw = input("-> > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(shown):
            return shown[int(raw) - 1]
        # Any non-number input (or 4) is treated as the user's custom answer.
        return raw


def generate_text(model, tokenizer, prompt: str, args, device) -> str:
    chunks: List[str] = []
    max_new_tokens = effective_generation_tokens(args)
    display = GenerationDisplay(
        show_thinking=bool(getattr(args, "show_thinking", False)),
        show_tool_calls=False,
    )
    for chunk in generate_streaming(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
        loss_chunk_size=args.loss_chunk_size,
        max_answer_tokens=getattr(args, "max_answer_tokens", None),
        max_think_tokens=getattr(args, "max_think_tokens", None),
    ):
        chunks.append(chunk)
        visible = display.put(chunk)
        if visible:
            print(visible, end="", flush=True)
    print()
    return "".join(chunks)


def run_agent(args) -> None:
    model, tokenizer, device, checkpoint_path = load_model_for_inference(args)
    history: List[Tuple[str, str]] = []
    memory = ""
    ctx_len = unwrap_model(model).config.max_position_embeddings
    agent_root = str(resolve_root(getattr(args, "agent_root", None)))
    dry_run_tools = bool(getattr(args, "dry_run_tools", False))
    tool_log = Path(getattr(args, "agent_tool_log", "logs/agent_tools.jsonl"))
    if not tool_log.is_absolute():
        tool_log = Path(agent_root) / tool_log
    plan_mode_enabled = not bool(getattr(args, "no_plan_mode", False))
    plan_system_prompt = getattr(args, "system_prompt", None) or AGENT_SYSTEM_PROMPT
    print(f"[Agent] Loaded {checkpoint_path} on {device}. Type exit to stop.")
    print(f"[Agent] tool_root={agent_root}, dry_run_tools={dry_run_tools}, log={tool_log}, plan_mode={plan_mode_enabled}")

    while True:
        try:
            user_input = input("\nTask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        history.append(("user", user_input))

        # --- Plan mode phase: plan first, clarify if needed, then wait for approval. ---
        plan_approved = False
        if plan_mode_enabled:
            plan_approved = run_plan_phase(model, tokenizer, args, device, history, memory, ctx_len, agent_root, dry_run_tools, tool_log)
            if not plan_approved:
                # User declined the plan or interrupted; wait for a new task.
                continue

        # --- Execution phase: tool loop. ---
        for _round in range(4):
            max_new_tokens = effective_generation_tokens(args)
            system_prompt = getattr(args, "system_prompt", None) or ""
            prompt, memory = render_chat_prompt(tokenizer, system_prompt, history, memory, ctx_len, max_new_tokens)
            print("Agent > ", end="", flush=True)
            answer = generate_text(model, tokenizer, prompt, args, device).strip()
            tool_call = extract_tool_call(answer)
            if tool_call is None:
                history.append(("assistant", answer))
                break

            history.append(("assistant", answer))
            if dry_run_tools:
                tool_result = run_tool(tool_call, root=agent_root, dry_run=True, log_path=tool_log)
            elif not confirm_tool(tool_call):
                tool_result = "User denied this tool call. Continue without executing it."
            else:
                tool_result = run_tool(tool_call, root=agent_root, dry_run=False, log_path=tool_log)
            print(f"[Tool result]\n{tool_result}\n")
            history.append(("tool", tool_result))


def run_plan_phase(
    model,
    tokenizer,
    args,
    device,
    history: List[Tuple[str, str]],
    memory: str,
    ctx_len: int,
    agent_root: str,
    dry_run_tools: bool,
    tool_log: Path,
) -> bool:
    """Drive the plan-first phase for a single task.

    Asks the model to produce a <PLAN> (or a clarifying <ASK>). If the model asks a
    clarifying question, present options 1..3 plus a custom option 4, feed the answer
    back, and loop until a plan is produced. Then show the plan and ask the user to
    approve it. On approval, save the plan to plan.md via the write_file tool and
    record the approval in history. Returns True when the plan was approved.
    """
    system_prompt = getattr(args, "system_prompt", None) or AGENT_SYSTEM_PROMPT
    for _ in range(6):  # bound the clarify <-> plan loop
        max_new_tokens = effective_generation_tokens(args)
        prompt, memory = render_chat_prompt(tokenizer, system_prompt, history, memory, ctx_len, max_new_tokens)
        print("Agent [plan] > ", end="", flush=True)
        answer = generate_text(model, tokenizer, prompt, args, device).strip()

        # Priority 1: a clarifying question to resolve ambiguity before planning.
        ask = extract_ask(answer)
        if ask:
            history.append(("assistant", answer))
            user_answer = prompt_for_answer(ask)
            if not user_answer:
                print("[Plan] Clarification cancelled.")
                return False
            history.append(("user", user_answer))
            continue

        # Priority 2: a plan to review.
        plan_body = extract_plan(answer)
        if plan_body:
            history.append(("assistant", answer))
            print("\n[Proposed plan]")
            print(format_plan_for_display(plan_body))
            try:
                approval = input("\nНачать выполнение? (yes/no) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if approval not in {"yes", "y", "да", "д"}:
                history.append(("user", "plan not approved, revise"))
                continue
            # Save the plan to plan.md through the write_file tool path.
            plan_result = save_plan_to_file(plan_body, root=agent_root, dry_run=dry_run_tools, log_path=tool_log)
            print(f"[Plan saved] {plan_result}")
            history.append(("tool", "plan approved"))
            return True

        # The model emitted neither a question nor a plan. Record it and stop the phase
        # so the user is not stuck in a loop.
        history.append(("assistant", answer))
        print("[Plan] No plan or clarifying question produced; proceeding without an approved plan.")
        return False

    print("[Plan] Too many clarification rounds; aborting plan phase.")
    return False


if __name__ == "__main__":
    from main import parse_args

    run_agent(parse_args())
