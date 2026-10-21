"""Refresh only reviewed source paths from an explicitly pinned Git commit."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "export-manifest.json"


def approved_path(name):
    if not isinstance(name, str):
        raise ValueError("export path must be a string")
    path = PurePosixPath(name)
    if ("\\" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or len(path.parts) > 2):
        raise ValueError(f"invalid export path {name!r}")
    if name in {"LICENSE", "requirements-policy.txt"}:
        return path
    if (len(path.parts) != 2 or (path.parts[0], path.suffix) not in
            {("tcg", ".py"), ("scripts", ".py"), ("experiments", ".json")}):
        raise ValueError(f"path is outside reviewed source roots: {name}")
    if name in {"scripts/sync_template.py", "scripts/verify_template.py"}:
        raise ValueError("source refresh cannot replace template controls")
    return path


def git(source, *arguments):
    return subprocess.check_output(["git", "-C", str(source), *arguments], stderr=subprocess.PIPE)


def refresh(source, commit, write=False):
    source = source.resolve(strict=True)
    source_root = Path(git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if source_root == ROOT or ROOT.is_relative_to(source_root) or source_root.is_relative_to(ROOT):
        raise ValueError("source and template must be separate repositories")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be an explicit full lowercase Git commit hash")
    if git(source, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("source commit did not resolve to the requested identity")
    manifest = json.loads(MANIFEST.read_bytes())
    if manifest.get("schema") != "tcg-rl-env/source-export/1" or not manifest.get("files"):
        raise ValueError("unsupported or empty export manifest")
    if (ROOT / ".git").exists() and git(ROOT, "status", "--porcelain", "--", "export-manifest.json").strip():
        raise ValueError("commit existing export-manifest changes before refreshing")
    updates = {}
    for name, expected in manifest["files"].items():
        approved_path(name)
        destination = ROOT / name
        if any(parent.is_symlink() for parent in (destination, *destination.parents)):
            raise ValueError(f"export destination contains a symlink: {name}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"export destination must be a file: {name}")
        if any(parent.exists() and not parent.is_dir() for parent in destination.parents):
            raise ValueError(f"export destination parent must be a directory: {name}")
        if destination.exists() and sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f"preserve local export edits before refreshing: {name}")
        if (ROOT / ".git").exists() and git(ROOT, "status", "--porcelain", "--", name).strip():
            raise ValueError(f"commit existing export edits before refreshing: {name}")
        tree = git(source, "ls-tree", commit, "--", name).decode().strip()
        if not tree or tree.split()[0] != "100644":
            raise ValueError(f"source must be a regular nonexecutable tracked file: {name}")
        updates[name] = git(source, "show", f"{commit}:{name}")
    changes = [name for name, data in updates.items()
               if not (ROOT / name).is_file() or (ROOT / name).read_bytes() != data]
    if write:
        temporary = MANIFEST.with_suffix(".pending")
        if temporary.exists():
            raise ValueError("preserve an existing pending manifest before refreshing")
        for name, data in updates.items():
            destination = ROOT / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        manifest["source"]["commit"] = commit
        manifest["files"] = {name: sha256(data).hexdigest() for name, data in updates.items()}
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(MANIFEST)
    return {"mode": "written" if write else "dry_run", "source_commit": commit,
            "reviewed_paths": len(updates), "changed_paths": changes,
            "new_source_paths_imported": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        result = refresh(args.source, args.commit, args.write)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"refresh rejected: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
