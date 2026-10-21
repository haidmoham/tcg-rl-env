"""Bounded native policy distillation. Importing starts no games or training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from contextlib import contextmanager


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed32(seed, domain, *identity):
    value = int.from_bytes(hashlib.sha256(canonical([seed, domain, *identity])).digest()[:4], "little")
    return value or 1


def validate_inputs(protocol_path, bindings_path, mode):
    protocol = json.loads(Path(protocol_path).read_bytes())
    bindings = json.loads(Path(bindings_path).read_bytes())
    if protocol.get("schema") != "86-tcg-agent/mew-student-stage1/1":
        raise ValueError("unsupported policy distillation protocol")
    if protocol.get("stage") != "policy_only_distillation":
        raise ValueError("this runner implements policy distillation only")
    if mode != "qualify" and protocol.get("execution_enabled") is not True:
        raise ValueError("the frozen protocol must explicitly enable execution")
    artifacts = bindings["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("source closure is missing")
    if hashlib.sha256(canonical(artifacts)).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("artifact closure digest differs")
    resolved = set()
    for source, expected in artifacts.items():
        path = Path(source).resolve(strict=True)
        if file_hash(path) != expected:
            raise ValueError(f"source closure mismatch for {path.name}")
        resolved.add(path)
    root = Path(__file__).resolve().parents[1]
    required = [Path(__file__), root / "tcg/native_policy.py", root / "scripts/train_mew_student.py"]
    required += [Path(bindings[key]) for key in ("teacher_checkpoint", "engine_library", "deck_manifest")]
    if any(path.resolve(strict=True) not in resolved for path in required):
        raise ValueError("executing sources, engine, checkpoint and decks must belong to the pinned closure")
    for key, hash_key in (("teacher_checkpoint", "teacher_sha256"), ("engine_library", "engine_sha256"),
                          ("deck_manifest", "deck_manifest_sha256")):
        if file_hash(bindings[key]) != bindings[hash_key]:
            raise ValueError(f"{key} identity differs")
    limits = protocol["limits"]
    maxima = {"wall_seconds": 7200, "admitted_learner_microsteps": 100000,
              "teacher_query_seconds": 1800, "teacher_queries": 25000,
              "max_game_microsteps": 4096, "per_game_wall_seconds": 120}
    for key, upper in maxima.items():
        if type(limits[key]) not in (int, float) or not 0 < limits[key] <= upper:
            raise ValueError(f"invalid {key} limit")
    if not 0 < limits["stage_wall_seconds"] <= min(1800, limits["wall_seconds"]):
        raise ValueError("stage 1 must preserve the declared later-stage wall allocation")
    if not 0 < limits["finalization_reserve_seconds"] < limits["stage_wall_seconds"]:
        raise ValueError("stage finalization reserve must fit within the stage allocation")
    learning = protocol["learning"]
    if not 0 < learning["query_probability"] <= 1 or learning["temperature"] != 1.0:
        raise ValueError("query probability or policy temperature differs")
    if not 0 < learning["lr"] <= 0.001:
        raise ValueError("invalid learning rate")
    for key in ("update_every_queried_choices", "max_updates"):
        if type(learning[key]) is not int or learning[key] <= 0:
            raise ValueError(f"invalid {key}")
    manifest = json.loads(Path(bindings["deck_manifest"]).read_bytes())
    decks = tuple(tuple(row["card_ids"]) for row in manifest["decks"])
    if len(decks) != 6 or any(len(deck) != 60 or any(type(card) is not int or card <= 0 for card in deck) for deck in decks):
        raise ValueError("this historical pilot requires six exact 60-card lists")
    if protocol["evaluation"]["games_per_checkpoint"] != 72:
        raise ValueError("evaluation requires 36 deck pairs with both learner seats")
    for value in protocol["seeds"].values():
        if type(value) is not int or not 0 < value < 2**32:
            raise ValueError("all declared seed roots must be nonzero uint32 values")
    for key in ("max_frames", "max_microsteps"):
        if type(protocol["probe"][key]) is not int or protocol["probe"][key] <= 0:
            raise ValueError("probe bounds must be positive integers")
    return protocol, bindings, decks


def configure_runtime(bindings):
    if os.name != "posix" or not hasattr(os, "nice"):
        raise RuntimeError("the CPU pilot requires Linux or WSL nice 19")
    current = os.nice(0)
    if current < 19:
        os.nice(19 - current)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["FASTENC_LIB"] = str(Path(bindings["engine_library"]).resolve())
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    from . import native_policy as native
    native.configure_mew(bindings["mew_root"])
    return torch, native


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.started = time.monotonic()
        self.admitted = 0
        self.teacher_queries = 0
        self.teacher_seconds = 0.0
        self.work_deadline = None

    def stop_reason(self):
        if time.monotonic() - self.started >= self.limits["wall_seconds"]:
            return "arm_wall_cap"
        if self.admitted >= self.limits["admitted_learner_microsteps"]:
            return "admitted_microstep_cap"
        if self.work_deadline is not None and time.monotonic() >= self.work_deadline:
            return "stage_finalization_reserve"
        return None

    def teacher_available(self):
        return (self.stop_reason() is None and self.teacher_queries < self.limits["teacher_queries"]
                and self.teacher_seconds < self.limits["teacher_query_seconds"])

    def receipt(self):
        return {"wall_seconds": time.monotonic() - self.started, "admitted_learner_microsteps": self.admitted,
                "teacher_queries": self.teacher_queries, "teacher_seconds": self.teacher_seconds}


@contextmanager
def wall_guard(seconds):
    def expired(signum, frame):
        raise TimeoutError("bounded native game wall timer expired")
    prior_handler = signal.signal(signal.SIGALRM, expired)
    existing = signal.getitimer(signal.ITIMER_REAL)
    if existing[0] > 0:
        seconds = min(seconds, existing[0])
    prior_timer = signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, prior_timer[0] - (time.monotonic() - started)), prior_timer[1])


def advice(torch, native, teacher, frame, budget):
    if not budget.teacher_available():
        return None
    started = time.monotonic()
    try:
        scores = native.score_policy(teacher, frame)
        probabilities = torch.softmax(scores.logits[0], -1).detach().clone()
        if not torch.isfinite(probabilities).all() or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0)):
            raise ValueError("invalid teacher distribution")
        return probabilities
    finally:
        budget.teacher_queries += 1
        budget.teacher_seconds += time.monotonic() - started


def append_json(path, row):
    with Path(path).open("ab") as stream:
        stream.write(canonical(row) + b"\n")


def evaluation_plan(seed):
    return tuple({"index": index, "learner_deck": left, "opponent_deck": right, "learner_seat": seat,
                  "engine_seed": seed32(seed, "stage1-eval-engine-v1", left, right, seat),
                  "learner_seed": seed32(seed, "stage1-eval-learner-v1", left, right, seat),
                  "opponent_seed": seed32(seed, "stage1-eval-opponent-v1", left, right, seat)}
                 for index, (left, right, seat) in enumerate((a, b, s) for a in range(6) for b in range(6) for s in (0, 1)))


def _run_episode(torch, native, student, decks, assignment, protocol, budget, *, teacher=None,
                 query_rng=None, label_capacity=0, probe=False, external_wall_remaining=None):
    from fastenc.games import Games
    capture = native.FrameCapture()
    learner_rng = torch.Generator(device="cpu").manual_seed(assignment["learner_seed"])
    opponent_rng = torch.Generator(device="cpu").manual_seed(assignment["opponent_seed"])
    absolute_decks = [None, None]
    seat = assignment["learner_seat"]
    absolute_decks[seat] = decks[assignment["learner_deck"]]
    absolute_decks[1 - seat] = decks[assignment["opponent_deck"]]
    started = time.monotonic()
    labels, choices, steps, genuine, forced_exposed = [], [], 0, 0, 0
    status, reason, winner = "censored", None, None
    stats = {}
    remaining_wall = budget.limits["wall_seconds"] - (time.monotonic() - budget.started)
    if external_wall_remaining is not None:
        remaining_wall = min(remaining_wall, external_wall_remaining)
    if budget.work_deadline is not None:
        remaining_wall = min(remaining_wall, budget.work_deadline - time.monotonic())
    with wall_guard(min(protocol["limits"]["per_game_wall_seconds"], remaining_wall)), Games(
            [absolute_decks], seed=assignment["engine_seed"], seeds=[assignment["engine_seed"]], threads=1) as games:
        try:
            while True:
                reason = budget.stop_reason()
                if reason:
                    break
                if time.monotonic() - started >= protocol["limits"]["per_game_wall_seconds"]:
                    reason = "game_wall_cap"
                    break
                if steps >= protocol["limits"]["max_game_microsteps"]:
                    reason = "game_microstep_cap"
                    break
                with games.encode() as encoded:
                    record = encoded.records[0]
                    if int(record["decisionIndex"]) >= protocol["limits"]["max_game_microsteps"]:
                        reason = "native_decision_index_cap"
                        break
                    if int(record["terminal"]) == 1:
                        winner = int(record["winner"])
                        if winner not in (0, 1, 2):
                            raise ValueError("native terminal has an unknown winner")
                        status, reason = "terminal", None
                        break
                    if encoded.batch is None:
                        raise ValueError("nonterminal game has no encodable actor")
                    frame = capture.capture(encoded)
                if len(frame.records) != 1:
                    raise ValueError("sequential collector expected exactly one decision row")
                rec = frame.records[0]
                legal = tuple(rec.eligible_columns)
                if not legal:
                    raise ValueError("native actor has no legal microchoice")
                learner_turn = rec.select_player == seat
                log_probability = None
                if learner_turn:
                    budget.admitted += 1
                    if len(legal) == 1:
                        forced_exposed += 1
                    else:
                        genuine += 1
                    scores = native.score_policy(student, frame)
                    action = int(torch.multinomial(torch.softmax(scores.logits[0], -1), 1, generator=learner_rng).item())
                    log_probability = float(torch.log_softmax(scores.logits[0], -1)[action])
                    query = (probe or (query_rng is not None and query_rng.random() < protocol["learning"]["query_probability"]))
                    if len(legal) > 1 and query and len(labels) < label_capacity and teacher is not None:
                        target = advice(torch, native, teacher, frame, budget)
                        if target is not None:
                            labels.append((frame, target))
                else:
                    action = legal[int(torch.randint(len(legal), (1,), generator=opponent_rng).item())]
                if action not in legal:
                    raise ValueError("sampled microchoice differs from native legal columns")
                choices.append({"frame_digest": frame.digest, "decision_index": rec.decision_index,
                                "prefix_len": rec.prefix_len, "actor_seat": rec.select_player,
                                "legal_count": len(legal), "action": action, "learner": learner_turn,
                                "student_log_probability": log_probability})
                if budget.stop_reason() == "arm_wall_cap":
                    reason = "arm_wall_cap_before_step"
                    break
                games.step([action])
                steps += 1
                if probe and len(labels) >= label_capacity:
                    reason = "probe_frame_target_reached"
                    break
        except TimeoutError as error:
            status, reason = "censored", str(error)
        except Exception as error:
            status, reason = "fault", f"{type(error).__name__}: {error}"
        stats = games.stats()
    row = dict(assignment, status=status, reason=reason, winner=winner,
               score=None if status != "terminal" else (0.5 if winner == 2 else float(winner == seat)),
               explicit_microsteps=steps, learner_genuine_choices=genuine, learner_exposed_forced_choices=forced_exposed,
               native_stats=stats, wall_seconds=time.monotonic() - started, labels=len(labels), choices=choices,
               microstep_definition="explicit learner-scored atomic choices; native auto-forced environment work separately reported")
    return row, labels, {"learner": learner_rng.get_state(), "opponent": opponent_rng.get_state()}


def run_episode(torch, native, student, decks, assignment, protocol, budget, **kwargs):
    started = time.monotonic()
    try:
        return _run_episode(torch, native, student, decks, assignment, protocol, budget, **kwargs)
    except Exception as error:
        timeout = isinstance(error, TimeoutError)
        return (dict(assignment, status="censored" if timeout else "fault",
                     reason=f"{type(error).__name__}: {error}", winner=None, score=None,
                     explicit_microsteps=None, learner_genuine_choices=None,
                     learner_exposed_forced_choices=None, native_stats=None, labels=0,
                     wall_seconds=time.monotonic() - started,
                     accounting="native construction, entry, exit or receipt collection did not complete"), [], {})


def label_metrics(torch, native, student, labels):
    rows = []
    for frame, target in labels:
        scores = native.score_policy(student, frame)
        legal = scores.legal_mask[0]
        q = target[legal]
        logp = torch.log_softmax(scores.logits[0][legal], -1)
        cross_entropy = -(q * logp).sum()
        entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()
        rows.append({"frame_digest": frame.digest, "cross_entropy": float(cross_entropy),
                     "forward_kl": float(cross_entropy - entropy),
                     "top1_agreement": int(scores.logits[0].argmax() == target.argmax())})
    return {"count": len(rows), "rows": rows,
            "mean_cross_entropy": sum(row["cross_entropy"] for row in rows) / len(rows) if rows else None,
            "mean_forward_kl": sum(row["forward_kl"] for row in rows) / len(rows) if rows else None,
            "top1_agreement": sum(row["top1_agreement"] for row in rows) / len(rows) if rows else None}


def fit_labels(torch, native, student, optimizer, labels, budget):
    if not labels:
        raise ValueError("cannot optimize an empty label batch")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for frame, target in labels:
        if budget.stop_reason():
            optimizer.zero_grad(set_to_none=True)
            return None
        scores = native.score_policy(student, frame, grad=True)
        legal = scores.legal_mask[0]
        loss = -(target[legal] * torch.log_softmax(scores.logits[0][legal], -1)).sum()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite distillation loss")
        (loss / len(labels)).backward()
        losses.append(float(loss.detach()))
    parameters = tuple(parameter for _, parameter in native.policy_parameters(student))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
    if budget.stop_reason():
        optimizer.zero_grad(set_to_none=True)
        return None
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in student.parameters()):
        raise ValueError("nonfinite updated student")
    return {"examples": len(labels), "cross_entropy": sum(losses) / len(losses), "gradient_norm": float(norm)}
