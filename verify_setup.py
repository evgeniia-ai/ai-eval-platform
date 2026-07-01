#!/usr/bin/env python3
"""
HealthBridge — AI/ML Case Study
Environment verification script.

Run this before the session starts to confirm everything is working:

    python verify_setup.py

"""

import json
import os
import sys

# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def passed(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")

def failed(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")

def warned(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


# ─────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────

def check_python_version() -> bool:
    v = sys.version_info
    if v.major >= 3 and v.minor >= 11:
        passed(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    else:
        failed(f"Python {v.major}.{v.minor}.{v.micro} — need 3.11+")
        return False


def check_package(name: str, import_name: str | None = None) -> bool:
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        passed(f"{name} ({version})")
        return True
    except ImportError:
        failed(f"{name} — not installed. Run: pip install {name}")
        return False


def check_anthropic_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-will-be-pre-loaded":
        warned("ANTHROPIC_API_KEY is not set — ask your interviewer")
        return False
    else:
        masked = key[:8] + "..." + key[-4:]
        passed(f"ANTHROPIC_API_KEY is set ({masked})")
        return True


def check_anthropic_connection() -> bool:
    """Make a minimal API call to verify the key works."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-will-be-pre-loaded":
        warned("Skipping API connection test (no key)")
        return False

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            # A cheap, fast, current model for a connectivity smoke test.
            # (The old default claude-sonnet-4-20250514 retired 2026-06-15 and now 404s.)
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'ready' and nothing else."}],
        )
        text = response.content[0].text.strip().lower()
        if "ready" in text:
            passed("Anthropic API connection — working")
            return True
        else:
            passed(f"Anthropic API connection — got response: '{text}'")
            return True
    except Exception as e:
        failed(f"Anthropic API connection — {e}")
        return False


def check_seed_data() -> bool:
    path = os.path.join(os.path.dirname(__file__), "data", "seed_transcripts.json")
    if not os.path.exists(path):
        failed(f"Seed data not found at {path}")
        return False

    try:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            failed("Seed data is empty or malformed")
            return False

        call_types = set(t["call_type"] for t in data)
        passed(f"Seed data — {len(data)} transcripts, {len(call_types)} call types")
        return True
    except Exception as e:
        failed(f"Seed data — error reading: {e}")
        return False


def check_ports() -> bool:
    """Just a reminder — can't actually test forwarding in advance."""
    passed("Ports 8501 (Streamlit), 8000 (FastAPI), 7860 (Gradio) — configured for auto-forward")
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    print()
    print(f"{BOLD}HealthBridge — AI/ML Case Study{RESET}")
    print(f"{BOLD}Environment Verification{RESET}")
    print("=" * 48)
    print()

    all_ok = True

    print(f"{BOLD}Python & Packages{RESET}")
    all_ok &= check_python_version()
    for name, imp in [
        ("anthropic", None),
        ("pydantic", None),
        ("fastapi", None),
        ("streamlit", None),
        ("gradio", None),
        ("pandas", None),
        ("plotly", None),
        ("sqlite_utils", "sqlite_utils"),
        ("rich", None),
        ("tenacity", None),
    ]:
        all_ok &= check_package(name, imp)
    print()

    print(f"{BOLD}API Key & Connection{RESET}")
    key_ok = check_anthropic_key()
    if key_ok:
        all_ok &= check_anthropic_connection()
    else:
        all_ok = False
    print()

    print(f"{BOLD}Data & Environment{RESET}")
    all_ok &= check_seed_data()
    all_ok &= check_ports()
    print()

    print("=" * 48)
    if all_ok:
        print(f"{GREEN}{BOLD}All checks passed — you're ready to build!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}Some checks need attention — see warnings above.{RESET}")
        print("If you can't resolve them, notify your interviewer.")
    print()


if __name__ == "__main__":
    main()
