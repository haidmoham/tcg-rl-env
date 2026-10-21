"""Bind local private dependencies for the historical native pilot without loading them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
HISTORICAL = {
    "engine_library": "b010aeea9f753a4dbb37040cc6d71b9e29f08114d519659097c37427887094b0",
    "teacher_checkpoint": "f3d38a0dafae01c748284ed39d9ba49f5923cde8cac3496a0fc3609c1e94e8dc",
    "deck_manifest": "93150420ccbd048c83f8d5c6c0123c27585dec8ea9996080aa737279b64e3f31",
}
HASH_FIELDS = {"engine_library": "engine_sha256", "teacher_checkpoint": "teacher_sha256",
               "deck_manifest": "deck_manifest_sha256"}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".cache", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", "node_modules", "build", "dist", "target",
    "runs", "logs", "checkpoints", "weights",
}
BUILD_SUFFIXES = {
    ".pyc", ".pyo", ".o", ".obj", ".a", ".lib", ".so", ".dll", ".dylib",
    ".exe", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
}
OWN_SOURCES = (
    "tcg/__init__.py", "tcg/native_policy.py", "tcg/native_distillation.py",
    "scripts/train_mew_student.py", "tcg/native_actorcritic.py",
    "scripts/train_native_actorcritic.py", "scripts/evaluate_native_pilot.py",
    "scripts/prepare_native_pilot_bindings.py",
    "tcg/native_sustained.py", "scripts/train_native_sustained.py",
    "scripts/evaluate_native_checkpoint.py", "scripts/run_native_campaign.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs)


def source_file(value):
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError("symlink inputs are ambiguous; supply an ordinary source file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("a required input is not a regular file")
    return path


def sha(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("an input changed while its identity was being read")
    return digest.hexdigest()


def private_sources(root):
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            if (current / name).is_symlink():
                raise ValueError("the private source tree contains a directory symlink")
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES)
        for name in sorted(files):
            path = current / name
            if path.is_symlink():
                raise ValueError("the private source tree contains a file symlink")
            if path.suffix.lower() not in BUILD_SUFFIXES and not name.endswith(("~", ".tmp", ".log")):
                yield source_file(path)


def prepare(args):
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("bindings output already exists")
    if not output.parent.is_dir():
        raise ValueError("the private output directory must already exist")
    if output.resolve().is_relative_to(PROJECT):
        raise ValueError("bindings must be written outside the project checkout")
    root = args.mew_root.expanduser().resolve(strict=True)
    if (not root.is_dir() or not (root / "agent-lab/imitation/model_v4.py").is_file()
            or not (root / "cabt-engine/fastenc/fastenc_export.cpp").is_file()):
        raise ValueError("Mew root must contain the complete agent-lab and cabt-engine source tree")
    paths = {key: source_file(getattr(args, key)) for key in HISTORICAL}
    for key in ("engine_library", "teacher_checkpoint"):
        if not paths[key].is_relative_to(root):
            raise ValueError("engine and teacher must belong to the selected private Mew root")
    for key, expected in HISTORICAL.items():
        if sha(paths[key]) != expected:
            raise ValueError(f"{key} does not match the historical v1 pilot identity")
    decks = read_json(paths["deck_manifest"])
    if len(decks.get("decks", ())) != 6 or any(len(row.get("card_ids", ())) != 60 for row in decks["decks"]):
        raise ValueError("historical v1 requires six complete 60-card deck lists")
    continuation_args = (args.stage1_run, args.stage1_bindings, args.stage1_protocol, args.actorcritic_protocol)
    continuation = all(value is not None for value in continuation_args)
    if any(value is not None for value in continuation_args) and not continuation:
        raise ValueError("continuation requires all four stage1-run/bindings/protocol and actorcritic-protocol inputs")
    artifacts, resolved = {}, {}

    def add(path, expected=None, original_key=None):
        path = source_file(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("an original stage-1 artifact changed")
        key = str(path) if original_key is None else original_key
        if path in resolved:
            previous_key = resolved[path]
            if artifacts[previous_key] != actual:
                raise ValueError("one resolved artifact has conflicting identities")
            if original_key is not None and previous_key != original_key:
                raise ValueError("the original artifact map contains ambiguous path aliases")
            return
        artifacts[key] = actual
        resolved[path] = key

    if continuation:
        original_path = source_file(args.stage1_bindings)
        original_protocol_path = source_file(args.stage1_protocol)
        actorcritic_path = source_file(args.actorcritic_protocol)
        original = read_json(original_path)
        if original.get("schema") != "mew-student-local-bindings/1":
            raise ValueError("unsupported original stage-1 bindings schema")
        original_map = original["artifacts"]
        if not original_map or hashlib.sha256(canonical(original_map)).hexdigest() != original["artifact_manifest_sha256"]:
            raise ValueError("original stage-1 artifact map has a mismatched digest")
        for key, expected in original_map.items():
            add(key, expected, original_key=key)
        for key, path in paths.items():
            if source_file(original[key]) != path or original[HASH_FIELDS[key]] != HISTORICAL[key]:
                raise ValueError("continuation changed the original engine, teacher or deck inputs")
        stage1 = args.stage1_run.expanduser().resolve(strict=True)
        if not stage1.is_dir():
            raise ValueError("stage1-run must be a completed run directory")
        manifest = read_json(source_file(stage1 / "manifest.json"))
        result = read_json(source_file(stage1 / "training-result.json"))
        initial = read_json(source_file(stage1 / "initial.json"))
        student = read_json(source_file(stage1 / "student.json"))
        protocol = read_json(original_protocol_path)
        protocol_digest = hashlib.sha256(canonical(protocol)).hexdigest()
        if (manifest.get("mode") != "train" or manifest["artifact_manifest_sha256"] != original["artifact_manifest_sha256"]
                or manifest["protocol"] != protocol or manifest["protocol_sha256"] != protocol_digest
                or initial["protocol_sha256"] != protocol_digest or student["protocol_sha256"] != protocol_digest):
            raise ValueError("stage-1 source, protocol and checkpoint lineage disagree")
        if (result.get("status") != "completed_bounded_stage" or result.get("critic_parameters_unchanged") is not True
                or student["weights_sha256"] != result["student_weights_sha256"]
                or initial["weights_sha256"] != result["initial_weights_sha256"]):
            raise ValueError("stage-1 completion and checkpoint receipts disagree")
        next_protocol = read_json(actorcritic_path)
        if (next_protocol.get("schema") != "86-tcg-agent/native-actorcritic/1"
                or next_protocol.get("sequence") != [3, 4] or next_protocol.get("skipped_stage") != 2
                or next_protocol.get("teacher_access") != "forbidden"):
            raise ValueError("continuation protocol must declare teacher-free stages 3 and 4 with stage 2 skipped")
        for name in ("manifest.json", "training-result.json", "initial.json", "student.json", "evaluation-plan.json"):
            add(stage1 / name)
        add(stage1 / "initial.pt", initial["weights_sha256"])
        add(stage1 / "student.pt", student["weights_sha256"])
        for name in ("initial.state.pt", "student.state.pt"):
            if (stage1 / name).exists():
                add(stage1 / name)
        for path in (original_path, original_protocol_path, actorcritic_path):
            add(path)
        bindings = {key: value for key, value in original.items() if key not in ("artifacts", "artifact_manifest_sha256")}
        bindings.update(stage1_bindings=str(original_path), stage1_bindings_sha256=sha(original_path),
                        stage1_protocol=str(original_protocol_path), stage1_protocol_sha256=sha(original_protocol_path))
    else:
        bindings = {"schema": "mew-student-local-bindings/1", "mew_root": str(root)}
    for path in private_sources(root):
        add(path)
    for path in paths.values():
        add(path)
    for name in OWN_SOURCES:
        path = PROJECT / name
        if path.exists():
            add(path)
        elif name in ("tcg/native_policy.py", "tcg/native_distillation.py", "scripts/train_mew_student.py"):
            raise ValueError("the project is missing a required stage-1 source")
    bindings.update({key: str(path) for key, path in paths.items()})
    bindings.update(engine_sha256=HISTORICAL["engine_library"], teacher_sha256=HISTORICAL["teacher_checkpoint"],
                    deck_manifest_sha256=HISTORICAL["deck_manifest"], mew_root=str(root))
    bindings["artifacts"] = dict(sorted(artifacts.items()))
    bindings["artifact_manifest_sha256"] = hashlib.sha256(canonical(bindings["artifacts"])).hexdigest()
    with output.open("xb") as stream:
        stream.write(canonical(bindings) + b"\n")
    return {"file": output.name, "sha256": sha(output), "artifacts": len(artifacts),
            "mode": "continuation" if continuation else "stage1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mew-root", type=Path, required=True)
    parser.add_argument("--engine-library", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--deck-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path)
    parser.add_argument("--stage1-bindings", type=Path)
    parser.add_argument("--stage1-protocol", type=Path)
    parser.add_argument("--actorcritic-protocol", type=Path)
    args = parser.parse_args()
    try:
        receipt = prepare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"binding preparation failed: {type(error).__name__}; no usable bindings produced\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
