"""Run the separately frozen sustained historical training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

INVOKED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg import native_sustained as sustained
from tcg.native_distillation import canonical, configure_runtime, file_hash, wall_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    parser.add_argument("--campaign-start-monotonic", type=float)
    args = parser.parse_args()
    origin = INVOKED if args.campaign_start_monotonic is None else args.campaign_start_monotonic
    if not math.isfinite(origin) or origin < 0 or origin > INVOKED:
        parser.error("campaign start must be a finite nonnegative same-host monotonic time no later than trainer invocation")
    args.output.mkdir(parents=True, exist_ok=False)
    budget, progress = None, sustained.TrainingProgress()
    try:
        protocol, bindings, decks = sustained.validate_inputs(args.protocol, args.bindings)
        if args.mode == "train" and protocol.get("execution_enabled") is not True:
            raise ValueError("sustained execution must be enabled in the frozen protocol")
        lineage = {"campaign_start_monotonic": origin, "trainer_invoked_monotonic": INVOKED,
                   "clock_origin": "standalone" if args.campaign_start_monotonic is None else "supervisor",
                   "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
                   "protocol_file_sha256": file_hash(args.protocol), "bindings_sha256": file_hash(args.bindings),
                   "source_artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                   "engine_sha256": bindings["engine_sha256"], "deck_manifest_sha256": bindings["deck_manifest_sha256"],
                   "teacher_sha256": bindings["teacher_sha256"], "parent_checkpoint_sha256": None}
        sustained.write_json(args.output / "manifest.json", {"protocol": protocol, "lineage": lineage,
                              "campaign_start_monotonic": origin,
                              "source_artifact_manifest": bindings["artifacts"],
                              "cadence": "External supervisor evaluates and reports only every 1800 seconds and at final endpoint. Completed update counts never trigger reports."})
        if args.mode == "validate":
            sustained.write_json(args.output / "validation.json", {"passed": True, "training_started": False})
            return 0
        budget = sustained.SustainedBudget(protocol, progress, started=origin, output=args.output)
        with wall_guard(max(0.001, budget.remaining(reserve=False))):
            torch, native = configure_runtime(bindings)
            torch.manual_seed(protocol["seeds"]["optimizer"])
            model = native.build_student(native.SmallNativeConfig(**protocol["student"]), seed=protocol["seeds"]["student_init"])
            initial = sustained.commit_checkpoint(torch, native, model, None, args.output, progress, budget, lineage, {})
            sustained.atomic_json(args.output / "initial-checkpoint.json", initial)
            lineage["initial_checkpoint_sha256"] = initial["weights_sha256"]
            lineage["parent_checkpoint_sha256"] = initial["weights_sha256"]
            sustained.distill(torch, native, model, decks, protocol, bindings, args.output, progress, budget, lineage)
        with wall_guard(max(0.001, min(protocol["limits"]["stage3_wall_seconds"],
                                     budget.started + protocol["limits"]["training_stop_campaign_seconds"] - time.monotonic()))):
            sustained.calibrate(torch, native, model, decks, protocol, args.output, progress, budget, lineage)
        with wall_guard(max(0.001, budget.started + protocol["limits"]["training_stop_campaign_seconds"] - time.monotonic())):
            result = sustained.improve(torch, native, model, decks, protocol, args.output, progress, budget, lineage)
        latest = json.loads((args.output / "latest-checkpoint.json").read_bytes())
        sustained.write_json(args.output / "training-result.json", {"status": "completed_bounded_run",
                              "progress": vars(progress), "latest_checkpoint": latest,
                              "resources": budget.receipt(), "stages": budget.stage_receipts,
                              "evaluation_required": True, "stage4": result})
        budget.publish_progress(force=True, status="completed_bounded_run")
        return 0
    except Exception as error:
        latest_path = args.output / "latest-checkpoint.json"
        latest = json.loads(latest_path.read_bytes()) if latest_path.is_file() else None
        bounded = isinstance(error, TimeoutError)
        sustained.write_json(args.output / "training-result.json", {
            "status": "bounded_stop" if bounded else "fault", "error": f"{type(error).__name__}: {error}",
            "progress": vars(progress), "latest_checkpoint": latest,
            "resources": budget.receipt() if budget else None,
            "checkpoint_policy": "Only the latest completed immutable receipt is usable. Any partially applied update is discarded and never published.",
            "evaluation_required": latest is not None})
        if budget is not None:
            budget.publish_progress(force=True, status="bounded_stop" if bounded else "fault")
        if not bounded:
            raise
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
