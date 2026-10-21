"""Verify exported bytes, offline contracts, and explicitly incomplete readiness."""

import argparse
import ast
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPLATE_FILES = {
    ".gitattributes", ".github/workflows/verify.yml", ".gitignore", "AGENTS.md",
    "README.md", "coverage/current.json", "export-manifest.json", "pyproject.toml",
    "scripts/sync_template.py", "scripts/verify_template.py",
}

from sync_template import approved_path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(arguments, expected=0):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=45)
    require(result.returncode == expected,
            f"{' '.join(arguments)} exited {result.returncode}: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


def exported_sources():
    manifest = json.loads((ROOT / "export-manifest.json").read_bytes())
    require(manifest.get("schema") == "tcg-rl-env/source-export/1", "unknown export schema")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["source"]["commit"]), "invalid source commit")
    require(manifest["source"]["license"] == "MIT", "source license must be retained")
    files = manifest["files"]
    require(isinstance(files, dict) and files, "empty source manifest")
    secret = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    private_path = re.compile(rb"(?:[A-Za-z]:[\\/](?:Users|86agent|Projects)[\\/]|/ho"
                              rb"me/[^/\s]+/|/mnt/[a-z]/)")
    for name, expected in files.items():
        approved_path(name)
        path = ROOT / name
        require(not any(parent.is_symlink() for parent in (path, *path.parents)),
                f"export path is a symlink: {name}")
        data = path.read_bytes()
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"invalid digest for {name}")
        require(sha256(data).hexdigest() == expected, f"export bytes changed: {name}")
    if (ROOT / ".git").exists():
        inventory = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT)
        actual = set(inventory.decode("utf-8").rstrip("\0").split("\0"))
    else:
        actual = set()
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if (relative.parts[0] in {"build", "dist", ".venv", "venv"}
                    or relative.parts[0].endswith(".egg-info") or "__pycache__" in relative.parts):
                continue
            if path.is_file() or path.is_symlink():
                actual.add(relative.as_posix())
    expected_files = set(files) | TEMPLATE_FILES
    require(actual == expected_files,
            f"publication inventory differs from allowlist: {sorted(actual ^ expected_files)}")
    for name in expected_files:
        path = ROOT / name
        require(not any(parent.is_symlink() for parent in (path, *path.parents)),
                f"publication path is a symlink: {name}")
        data = path.read_bytes()
        require(not secret.search(data), f"possible credential in {name}")
        require(not private_path.search(data), f"private machine path in {name}")
        if path.suffix == ".py":
            ast.parse(data, filename=name)
    return len(files)


def coverage_summary():
    summary = json.loads((ROOT / "coverage/current.json").read_bytes())
    require(summary.get("schema") == "tcg-rl-env/coverage-summary/1", "unknown coverage schema")
    require(summary.get("fullStandardReady") is False, "an aggregate summary cannot qualify full Standard")
    require(summary.get("blockers"), "incomplete coverage must expose blockers")
    for field in ("sourceReportSha256", "aggregateReceiptSha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", summary[field]) is not None, f"invalid {field}")
    denominator = summary["denominator"]
    for field, total in (("caseStatusCounts", "requiredCases"),
                         ("rowStatusCounts", "eligiblePrintings"),
                         ("programStatusCounts", "programs")):
        counts = summary[field]
        require(all(type(value) is int and value >= 0 for value in counts.values()), f"invalid {field}")
        require(sum(counts.values()) == denominator[total], f"coverage total does not reconcile: {field}")
    require(denominator["rawSourcePrintings"] + denominator["sourceMissingPrintings"]
            == denominator["eligiblePrintings"], "missing printings escaped the denominator")
    return summary


def decision_contract():
    from tcg.decision import ActorDecision, LegalPrompt, Option, SUBMIT, Token
    from tcg.rollout import Episode, EpisodeEnd, EpisodeProvenance, LearnerStep, admit_rollout

    options = tuple(Option(name, Token(role="option")) for name in ("first", "second"))
    decision = ActorDecision((), LegalPrompt(options, 2, 2, ordered=True))
    require(SUBMIT not in decision.choices(), "incomplete ordered selection must mask Submit")
    selected = decision.pick("second").pick("first")
    require(selected.choices() == (SUBMIT,), "complete selection must mask extra picks")
    require(selected.response() == ("second", "first"), "ordered selection lost order")
    try:
        selected.pick("second")
    except ValueError:
        pass
    else:
        raise ValueError("duplicate selection was accepted")
    provenance = EpisodeProvenance("complete", 0, "a" * 64, 0, "b" * 64)
    terminal = Episode(provenance, (LearnerStep(provenance, 0, decision, "first", -0.5, 0.0, 1.0),), EpisodeEnd.TERMINAL)
    truncated = Episode(replace(provenance, episode_id="truncated"), (), EpisodeEnd.TRUNCATION)
    targets = admit_rollout((terminal, truncated))
    require(targets.returns == (1.0,) and targets.censored_episode_ids == ("truncated",),
            "truncated episodes must remain censored and supply no return target")
    failed = Episode(replace(provenance, episode_id="failed"), (), EpisodeEnd.INFRASTRUCTURE)
    try:
        admit_rollout((terminal, failed))
    except ValueError:
        pass
    else:
        raise ValueError("infrastructure failure did not invalidate the rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full-standard", action="store_true")
    args = parser.parse_args()
    try:
        count = exported_sources()
        summary = coverage_summary()
        decision_contract()
        planner = command(["scripts/plan_campaign.py"])
        require(planner["execution_authorized"] is False, "planner must not authorize a campaign")
        readiness = command(["scripts/check_readiness.py"], expected=2)
        require(readiness["ready_to_train"] is False, "missing private qualification must block training")
        evaluation = command(["scripts/evaluate_native_checkpoint_288.py", "--protocol",
                              "experiments/mew-sustained-v1.json", "--amendment",
                              "experiments/mew-evaluation-288-v1.json", "--plan-only"])
        require(evaluation["games"] == 288 and evaluation["clusters"] == 144,
                "historical evaluation assignment count changed")
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"template verification failed: {error}\n")
    result = {"template": "verified", "exported_files": count,
              "checks": ["source_hashes", "source_ast", "publication_hygiene", "coverage_accounting",
                         "ordered_legal_selection", "rollout_outcomes", "nonexecuting_planner",
                         "expected_blocked_readiness", "288_game_evaluation_plan"],
              "full_standard": "blocked", "matched_declared_requirements": summary["caseStatusCounts"]["oracle_matched"],
              "total_declared_requirements": summary["denominator"]["requiredCases"],
              "scope": "Offline source, contract and receipt checks. No engine, model, games or training executed."}
    if args.require_full_standard:
        result["blockers"] = ["This gate version cannot certify full Standard. A reviewed independent admission validator is required.", *summary["blockers"]]
    print(json.dumps(result, indent=2))
    return 2 if args.require_full_standard else 0


if __name__ == "__main__":
    raise SystemExit(main())
