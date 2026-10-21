"""Explicit CABT v1 battle boundary. Importing this module loads no native code."""

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


class UnsupportedCapability(RuntimeError):
    pass


class EngineError(RuntimeError):
    pass


class _StartData(ctypes.Structure):
    _fields_ = [("pointer", ctypes.c_void_p), ("error_player", ctypes.c_int),
                ("error_type", ctypes.c_int)]


class _SerialData(ctypes.Structure):
    _fields_ = [("json", ctypes.c_char_p), ("data", ctypes.c_void_p),
                ("count", ctypes.c_int), ("actor", ctypes.c_int)]


@dataclass(frozen=True)
class EngineFrame:
    actor: int
    wire_observation: dict

    @property
    def result(self):
        return self.wire_observation["current"]["result"]


class CabtBattle:
    """One explicit library and battle per process; no implicit latest-version import."""

    _owner = None
    _libraries = {}

    def __init__(self, library_path, expected_sha256, *, seed_api=None):
        if seed_api is not None and (type(seed_api) is not str or seed_api != "BattleStartSeeded64"):
            raise ValueError("unsupported CABT seed API declaration")
        path = Path(library_path).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("CABT artifact hash mismatch")
        if CabtBattle._owner is not None:
            raise RuntimeError("use a separate worker process for another CABT battle")
        library_key = os.path.normcase(str(path))
        cached = self._libraries.get(library_key)
        if cached is not None and cached[0] != actual:
            raise RuntimeError("CABT artifact changed at a loaded path; restart the worker process")
        if cached is None:
            library = self._load_library(path)
            self._libraries[library_key] = (actual, library)
        else:
            library = cached[1]
        seed_start = None
        if seed_api is not None:
            try:
                seed_start = getattr(library, seed_api)
            except AttributeError as error:
                raise UnsupportedCapability("declared CABT seed API is absent from this artifact") from error
            seed_start.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint64]
            seed_start.restype = _StartData
        self._library = library
        self._seed_start = seed_start
        self._seed_api = seed_api
        self._pointer = None
        self._closed = False
        self.library_sha256 = actual
        CabtBattle._owner = self

    @property
    def seed_api(self):
        """Declared ABI only. Native determinism requires independent qualification."""
        return self._seed_api

    @staticmethod
    def _load_library(path):
        library = ctypes.CDLL(str(path))
        library.GameInitialize.argtypes = []
        library.GameInitialize.restype = None
        library.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.BattleStart.restype = _StartData
        library.GetBattleData.argtypes = [ctypes.c_void_p]
        library.GetBattleData.restype = _SerialData
        library.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        library.Select.restype = ctypes.c_int
        library.BattleFinish.argtypes = [ctypes.c_void_p]
        library.BattleFinish.restype = None
        library.AllCard.argtypes = []
        library.AllCard.restype = ctypes.c_char_p
        library.GameInitialize()
        return library

    def card_ids(self):
        if self._closed:
            raise RuntimeError("battle boundary is closed")
        cards = json.loads(self._library.AllCard().decode("utf-8"))
        ids = tuple(card["cardId"] for card in cards)
        if any(type(card) is not int or card < 0 for card in ids) or len(set(ids)) != len(ids):
            raise EngineError("invalid card catalog IDs")
        return ids

    def start(self, deck0, deck1, *, engine_seed=None):
        if engine_seed is not None:
            if type(engine_seed) is not int or not 0 <= engine_seed < 2**64:
                raise ValueError("engine seed must be an unsigned 64-bit integer")
            if self._seed_start is None:
                raise UnsupportedCapability("CABT v1 exposes no authoritative engine seed API by default")
        if self._closed or self._pointer is not None:
            raise RuntimeError("start requires a fresh open battle")
        for deck in (deck0, deck1):
            if len(deck) != 60 or any(type(card) is not int or card < 0 for card in deck):
                raise ValueError("each deck must contain exactly 60 nonnegative integer card IDs")
        cards = (ctypes.c_int * 120)(*(tuple(deck0) + tuple(deck1)))
        started = (self._library.BattleStart(cards) if engine_seed is None
                   else self._seed_start(cards, engine_seed))
        if not started.pointer:
            raise EngineError(f"deck admission failed for player {started.error_player}: {started.error_type}")
        self._pointer = started.pointer
        return self.observe()

    def observe(self):
        if self._closed or self._pointer is None:
            raise RuntimeError("no active battle")
        serial = self._library.GetBattleData(self._pointer)
        if not serial.json:
            raise EngineError("engine returned no JSON observation")
        wire = json.loads(serial.json.decode("utf-8"))
        if not isinstance(wire, dict) or not {"logs", "current", "select"} <= wire.keys():
            raise EngineError("unsupported CABT observation structure")
        # The separate native search buffer is privileged and never crosses this boundary.
        observation = {key: wire[key] for key in ("logs", "current", "select")}
        return EngineFrame(serial.actor, observation)

    def submit(self, indices):
        if self._closed or self._pointer is None:
            raise RuntimeError("no active battle")
        if not isinstance(indices, (list, tuple)) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("response must contain nonnegative integer option indices")
        selected = (ctypes.c_int * len(indices))(*indices)
        error = self._library.Select(self._pointer, selected, len(indices))
        if error:
            raise EngineError(f"CABT rejected response with code {error}")
        return self.observe()

    def close(self):
        if self._closed:
            return
        if self._pointer is not None:
            self._library.BattleFinish(self._pointer)
            self._pointer = None
        self._closed = True
        if CabtBattle._owner is self:
            CabtBattle._owner = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
