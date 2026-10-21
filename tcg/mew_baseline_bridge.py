"""Versioned actor-visible Mew wire projection for the historical CABT policy."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform

import torch

from . import cabt_observation as cabt
from .decision import ActorDecision, SUBMIT
from .episode_runner import FrozenPolicy
from .policy import Policy, PolicyConfig
from .training_checkpoint import TrainingCounters, TrainingIdentity


BRIDGE_SCHEMA = "mew-to-cabt-actor/2"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class InferenceTransport:
    """Integrity-checked weight transport, requiring separate forward parity evidence."""

    policy: Policy
    weights_sha256: str
    receipt: dict
    original_state: dict

    def assert_unchanged(self):
        if (any(module.training for module in self.policy.modules())
                or any(parameter.requires_grad for parameter in self.policy.parameters())
                or any(value.device.type != "cpu" or value.dtype != torch.float32
                       or not torch.equal(value, self.original_state[name])
                       for name, value in self.policy.state_dict().items())):
            raise ValueError("transported inference policy changed")


def transport_inference_checkpoint(directory, vocabulary, expected_weights_sha256):
    """Keep historical runtime strict loaders intact; transport inference weights only."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("checkpoint directory must not be a symlink")
    files = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("invalid checkpoint entry")
        files[path.name] = path.read_bytes()
    manifest = json.loads(files["manifest.json"])
    config = PolicyConfig()
    if "payload" in manifest:
        if set(manifest) != {"payload", "sha256"}:
            raise ValueError("invalid trained envelope")
        payload = manifest["payload"]
        if sha256(_json_bytes(payload)).hexdigest() != manifest["sha256"]:
            raise ValueError("trained manifest integrity mismatch")
        expected_fields = {"schema", "provenance", "identity", "counters", "model_config", "vocabulary",
                           "weights_sha256", "files", "python_version", "torch_version", "in_flight_games",
                           "partial_minibatch", "native_chance_exact_resume", "resume_scope", "platform", "cpu_threads"}
        if (set(payload) != expected_fields or payload["schema"] != "86-tcg-agent/trained-baseline/1"
                or payload["provenance"] != "random-origin-terminal-outcome-ppo"
                or payload["in_flight_games"] != 0 or payload["partial_minibatch"] is not False
                or payload["native_chance_exact_resume"] is not False):
            raise ValueError("unsupported trained checkpoint boundary")
        TrainingIdentity(**payload["identity"])
        TrainingCounters(**payload["counters"])
        if set(payload["files"]) != {"weights.pt", "optimizer.pt", "population.json", "samplers.json"}:
            raise ValueError("invalid trained package members")
        if set(files) != {*payload["files"], "manifest.json"} or any(
                sha256(files[name]).hexdigest() != digest for name, digest in payload["files"].items()):
            raise ValueError("trained package integrity mismatch")
        model_config = payload["model_config"]
        source_runtime = {key: payload[key] for key in ("python_version", "torch_version", "platform", "cpu_threads")}
    else:
        if set(files) != {"weights.pt", "manifest.json"}:
            raise ValueError("invalid initial package members")
        required = {"schema", "initialization", "optimizer", "seed", "torch_version", "config", "vocabulary",
                    "weights_sha256", "manifest_sha256"}
        if set(manifest) != required:
            raise ValueError("invalid initial manifest")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if (sha256(_json_bytes(payload)).hexdigest() != manifest["manifest_sha256"]
                or payload["schema"] != "86-tcg-agent/untrained-policy/1"
                or payload["initialization"] != "random" or payload["optimizer"] is not None):
            raise ValueError("invalid initial provenance or integrity")
        model_config = payload["config"]
        source_runtime = {"torch_version": payload["torch_version"]}
    if (_json_bytes(model_config) != _json_bytes(asdict(config))
            or _json_bytes(payload["vocabulary"]) != _json_bytes(asdict(vocabulary))
            or payload["weights_sha256"] != expected_weights_sha256
            or sha256(files["weights.pt"]).hexdigest() != expected_weights_sha256):
        raise ValueError("transport identity, architecture or vocabulary mismatch")
    state = torch.load(BytesIO(files["weights.pt"]), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        policy = Policy(vocabulary, config).cpu().float().eval().requires_grad_(False)
    expected = policy.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected) or any(
            type(value) is not torch.Tensor or value.shape != expected[name].shape
            or value.dtype != torch.float32 or not torch.isfinite(value).all() for name, value in state.items()):
        raise ValueError("transport weight tensor schema mismatch")
    policy.load_state_dict(state, strict=True)
    if any(path.read_bytes() != files[path.name] for path in directory.iterdir()):
        raise ValueError("checkpoint changed during transport")
    receipt = {"schema": "cabt-inference-runtime-transport/1", "original_runtime": source_runtime,
               "inference_runtime": {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
                                     "platform": platform.system(), "cpu_threads": torch.get_num_threads()},
               "weights_sha256": expected_weights_sha256, "manifest_sha256": sha256(files["manifest.json"]).hexdigest(),
               "strict_original_runtime_loader": False, "optimizer_restored": False,
               "forward_parity_required": {"rtol": 1e-5, "atol": 1e-6, "exact_mask_ids_and_greedy": True}}
    return InferenceTransport(policy, expected_weights_sha256, receipt,
                              {name: value.clone() for name, value in policy.state_dict().items()})


def _fields(value, old, added, path):
    if type(value) is not dict or set(value) - set(old) - set(added):
        raise ValueError(f"unsupported {path} fields")
    for key, expected in added.items():
        if key in value and (type(value[key]) is not type(expected) or value[key] != expected):
            raise ValueError(f"unsupported active mechanic {path}.{key}")
    return {key: value[key] for key in old if key in value}


def _cards(values, path, pokemon=False):
    if values is None:
        return None
    if type(values) is not list:
        raise ValueError(f"invalid {path} list")
    result = []
    for index, value in enumerate(values):
        location = f"{path}[{index}]"
        if value is None:
            result.append(None)
            continue
        added = {"teamPlasma": False}
        if pokemon:
            added["imprisonMarker"] = False
            provisions = value.get("energyProvisions", [])
            if type(provisions) is not list:
                raise ValueError(f"invalid {location}.energyProvisions")
            serials = {card["serial"] for card in value.get("energyCards", [])}
            for provision in provisions:
                if (type(provision) is not dict or set(provision) != {"serial", "mask", "count"}
                        or any(type(number) is not int or number < 0 for number in provision.values())
                        or provision["serial"] not in serials):
                    raise ValueError(f"unsupported {location}.energyProvisions")
            value = {key: item for key, item in value.items() if key != "energyProvisions"}
        else:
            added["opponentFacingTwoSidedStadiumSide"] = False
        projected = _fields(value, cabt.POKEMON_FIELDS if pokemon else cabt.CARD_FIELDS, added, location)
        if pokemon:
            for key in ("energyCards", "tools", "preEvolution"):
                projected[key] = _cards(projected[key], location + "." + key)
        result.append(projected)
    return result


def project_mew_observation(wire: dict) -> dict:
    """Drop privileged extensions before the historical adapter sees the frame.

    Energy provisions are an attachment summary. The original energies and attached
    card identities remain unchanged. Added mechanics without an old representation
    must remain inactive. This projection does not qualify the native redaction.
    """
    allowed = {"current", "select", "logs", "search_begin_input"}
    if type(wire) is not dict or set(wire) - allowed:
        raise ValueError("unsupported observation fields")
    raw_current = wire["current"]
    if type(raw_current) is not dict:
        raise ValueError("invalid current")
    current = _fields({key: value for key, value in raw_current.items() if key != "encoderState"},
                      cabt.STATE_FIELDS, {"enemyActiveNoAbilityThisTurn": False}, "current")
    players = current["players"]
    if type(players) is not list or len(players) != 2:
        raise ValueError("expected two players")
    actor = current["yourIndex"]
    if type(actor) is not int or actor not in (0, 1):
        raise ValueError("invalid actor")
    current["players"] = []
    for seat, raw_player in enumerate(players):
        sleep_count = raw_player.get("sleepCoinCount", 0)
        if type(sleep_count) is not int or sleep_count not in (0, 1):
            raise ValueError("unsupported multi-coin sleep mechanic")
        player = _fields({key: value for key, value in raw_player.items() if key != "sleepCoinCount"}, cabt.PLAYER_FIELDS, {
            "ownersBirthdayToday": False, "trainerCoinNullNextTurn": False,
            "toolAndStadiumSuppressed": False, "lostZone": [],
        }, f"players[{seat}]")
        for key in ("active", "bench", "discard", "hand"):
            player[key] = (_cards(player[key], f"players[{seat}].{key}", key in ("active", "bench"))
                           if key != "hand" or seat == actor else None)
        if type(player["prize"]) is not list:
            raise ValueError("invalid prize list")
        player["prize"] = [None] * len(player["prize"])
        current["players"].append(player)
    for key in ("stadium", "looking"):
        current[key] = _cards(current[key], key)
    select = _fields(wire["select"], cabt.SELECT_FIELDS, {
        "distinctCardCategories": False, "distinctCardNames": False, "distinctPokemonTypes": False,
    }, "select")
    for key, maximum in (("type", 11), ("context", 49)):
        if type(select[key]) is not int or not 0 <= select[key] <= maximum:
            raise ValueError(f"unsupported select.{key}")
    select["deck"] = _cards(select["deck"], "select.deck")
    for key in ("contextCard", "effect"):
        if select[key] is not None:
            select[key] = _cards([select[key]], f"select.{key}")[0]
    logs = wire["logs"]
    if type(logs) is not list:
        raise ValueError("invalid logs")
    projected_logs = []
    for event in logs:
        if type(event) is dict and "area" in event:
            if event.get("type") != 0 or type(event["area"]) is not int or event["area"] != 1:
                raise ValueError("unsupported non-deck shuffle log")
            event = {key: value for key, value in event.items() if key != "area"}
        if type(event) is dict and "prizeIndex" in event:
            index = event["prizeIndex"]
            if (type(index) is not int or not 0 <= index < 6 or event.get("type") not in (6, 7)
                    or 6 not in (event.get("fromArea"), event.get("toArea"))):
                raise ValueError("unsupported prize log slot")
            event = {key: value for key, value in event.items() if key != "prizeIndex"}
        if (type(event) is not dict or set(event) - set(cabt.LOG_FIELDS)
                or type(event.get("type")) is not int or not 0 <= event["type"] < 24):
            raise ValueError(f"unsupported log fields or kind: {event!r}")
        projected_logs.append(dict(event))
    return {"current": current, "select": select, "logs": projected_logs}


class GreedyBaselineActor:
    """Frozen CPU policy over exact historical microchoice masks."""

    def __init__(self, policy: FrozenPolicy | InferenceTransport):
        if type(policy) not in (FrozenPolicy, InferenceTransport):
            raise TypeError("FrozenPolicy or labeled InferenceTransport required")
        self.policy = policy
        self.forward_calls = 0
        self.forced_microsteps = 0
        self.microsteps = 0
        self.assert_unchanged()

    def assert_unchanged(self):
        if type(self.policy) is FrozenPolicy:
            self.policy.assert_identity(self.policy.weights_sha256, self.policy.policy.vocabulary)
        else:
            self.policy.assert_unchanged()

    def choose(self, decision: ActorDecision) -> tuple[int, ...]:
        if type(decision) is not ActorDecision:
            raise TypeError("actor receives only ActorDecision")
        while True:
            choices = decision.choices()
            if not choices:
                raise ValueError("empty legal choice set")
            with torch.inference_mode():
                output = self.policy.policy((decision,))
            self.forward_calls += 1
            self.microsteps += 1
            self.forced_microsteps += len(choices) == 1
            ids = output.option_ids[0]
            if (ids != (*tuple(option.option_id for option in decision.prompt.options), SUBMIT)
                    or output.logits.shape != (1, len(ids)) or output.values.shape != (1,)):
                raise ValueError("policy output differs from the declared option structure")
            expected = tuple(option_id in choices for option_id in ids)
            if tuple(output.legal_mask[0].tolist()) != expected:
                raise ValueError("policy mask differs from exact decision choices")
            if (not bool(torch.isfinite(output.logits[output.legal_mask]).all())
                    or not bool(torch.isneginf(output.logits[~output.legal_mask]).all())
                    or not bool(torch.isfinite(output.values).all())):
                raise ValueError("nonfinite legal policy output or unmasked illegal option")
            chosen = ids[int(output.logits[0].argmax().item())]
            if chosen not in choices:
                raise ValueError("policy chose an illegal microaction")
            if chosen == SUBMIT:
                return tuple(int(option_id) for option_id in decision.response())
            decision = decision.pick(chosen)
