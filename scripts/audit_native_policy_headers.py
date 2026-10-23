"""Compile and audit generated policy headers against pinned catalog inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def unknown_ids(known):
    values = sorted(known)
    candidates = {-(2**31), 2**31 - 1, -1, 1}
    candidates.update(left + 1 for left, right in zip(values, values[1:]) if left + 1 < right)
    return sorted(candidates - known)


def expected_inputs(args):
    catalog, identity, facts_receipt, export = (read(path) for path in
        (args.catalog, args.identity_receipt, args.facts_receipt, args.engine_receipt))
    for key, path in (("catalog", args.catalog), ("facts", args.facts), ("enums", args.enums),
                      ("identity_receipt", args.identity_receipt)):
        if sha(path) != facts_receipt["input_sha256"][key]:
            raise ValueError(f"facts receipt input differs: {key}")
    for path in (args.facts, args.enums):
        if sha(path) != export["outputs"][path.name]["sha256"]:
            raise ValueError("raw facts or enum export identity differs")
    identity_header = args.facts_receipt.parent / "native_identities.h"
    facts_header = args.facts_receipt.parent / "native_policy_facts.h"
    if (sha(args.catalog) != identity["catalog_sha256"]
            or sha(identity_header) != identity["header_sha256"]
            or sha(args.identity_receipt.parent / "native_identities.h") != identity["header_sha256"]
            or sha(identity_header) != facts_receipt["identity_header_sha256"]
            or sha(facts_header) != facts_receipt["facts_header_sha256"]):
        raise ValueError("generated headers differ from their declared input lineage")
    maps = {}
    for namespace in ("card", "attack", "skill"):
        table = catalog["catalog_maps"][namespace]
        inverse = table["dense_to_native"]
        if (not inverse or inverse[0] != 0 or len(set(inverse)) != len(inverse)
                or any(type(value) is not int or not -(2**31) <= value < 2**31 for value in inverse)):
            raise ValueError("catalog inverse is not an int32 bijection with reserved zero")
        expected = {str(native): dense for dense, native in enumerate(inverse)}
        if table["native_to_dense"] != expected or table["embedding_rows"] != len(inverse):
            raise ValueError("catalog forward/inverse mapping differs")
        maps[namespace] = [(native, dense) for dense, native in enumerate(inverse)]
    enums = read(args.enums)["CardType"]
    categories = {"Pokemon": "Pokemon", "Item": "Other", "Supporter": "Other", "Stadium": "Other",
                  "Tool": "Tool", "BasicEnergy": "Energy", "SpecialEnergy": "Energy"}
    if set(enums) != set(categories) or len(set(enums.values())) != len(enums):
        raise ValueError("native card enum requires an explicit category contract")
    by_type = {enums[name]: value for name, value in categories.items()}
    facts = read(args.facts)
    by_id = {row["cardId"]: row for row in facts}
    if len(by_id) != len(facts) or set(by_id) != {native for native, dense in maps["card"] if native}:
        raise ValueError("raw facts do not exactly cover catalog card identities")
    expected_facts, counts = [], Counter()
    for native, dense in maps["card"]:
        if native == 0:
            expected_facts.append((native, "Other", 0.0))
            continue
        row = by_id[native]
        raw = row["nativeFacts"]
        if raw["cardId"] != native or raw["cardType"] not in by_type:
            raise ValueError("raw native card identity/category differs")
        category = by_type[raw["cardType"]]
        baseline = raw["baselinePrizeValue"] if category == "Pokemon" else 0
        if type(baseline) is not int or baseline < 0:
            raise ValueError("native baseline prize count differs")
        prize = baseline / 2.0
        if (row["category"] != category.lower() or row["printedPrizeCount"] != baseline
                or row["printedPrizeValue"] != prize):
            raise ValueError("export projection disagrees with raw native facts")
        expected_facts.append((native, category, prize))
        counts[category] += 1
    if dict(counts) != facts_receipt["native_categories"]:
        raise ValueError("facts receipt category totals differ from independent derivation")
    paths = (args.catalog, args.identity_receipt, args.facts_receipt, args.engine_receipt,
             args.facts, args.enums, identity_header, facts_header,
             args.identity_receipt.parent / "native_identities.h")
    return maps, expected_facts, dict(counts), {str(path): sha(path) for path in paths}


def program(maps, facts):
    lines = ['#include "native_policy_facts.h"', '#include <type_traits>', '#include <iostream>',
             'using namespace native_policy_ids;',
             'static_assert(!std::is_convertible_v<NativeId<CardTag>,NativeId<AttackTag>>);',
             'static_assert(!std::is_convertible_v<DenseId<CardTag>,DenseId<SkillTag>>);',
             'static_assert(!std::is_convertible_v<NativeId<SkillTag>,DenseId<SkillTag>>);',
             'int main() { std::size_t identities=0, unknowns=0, negative_skills=0, facts=0;']
    for namespace, pairs in maps.items():
        title = namespace.title()
        lines.append(f'const int32_t {namespace}_expected[][2]={{')
        lines.extend(f'{{{native},{dense}}},' for native, dense in pairs)
        lines.extend(['};', f'for(const auto& pair:{namespace}_expected) {{',
             f'const NativeId<{title}Tag> native{{pair[0]}};',
             'const auto dense=ToDense(native);',
             'if(!dense || dense->value!=pair[1] || IsRegistered(native)!=(pair[0]!=0)) return 10;',
             '++identities;'])
        if namespace == "skill":
            lines.append('if(pair[0]<0) ++negative_skills;')
        lines.append('}')
        unknown = unknown_ids({native for native, dense in pairs})
        lines.append(f'const int32_t {namespace}_unknown[]={{' + ','.join(map(str, unknown)) + '};')
        lines.extend([f'for(const auto value:{namespace}_unknown) {{',
            f'const NativeId<{title}Tag> native{{value}};',
            'if(ToDense(native) || IsRegistered(native)) return 11;', '++unknowns;'])
        if namespace == "card":
            lines.append('if(native_policy_facts::ForCard(native)) return 12;')
        lines.append('}')
    lines.extend(['struct ExpectedFacts { int32_t native; native_policy_facts::Category category; float prize; };',
                  'const ExpectedFacts expected_facts[]={'])
    lines.extend(f'{{{native},native_policy_facts::Category::{category},{prize:.1f}f}},'
                 for native, category, prize in facts)
    lines.extend(['};', 'for(const auto& expected:expected_facts) {',
        'const auto actual=native_policy_facts::ForCard(NativeId<CardTag>{expected.native});',
        'if(!actual || actual->category!=expected.category || actual->baseline_prize_value!=expected.prize) return 13;',
        '++facts;', '}',
        'std::cout << "{\\"identities\\":" << identities << ",\\"unknowns\\":" << unknowns',
        '<< ",\\"negative_skills\\":" << negative_skills << ",\\"facts\\":" << facts << "}\\n";',
        'return 0;', '}'])
    return "\n".join(lines) + "\n"


def command_run(command, output, name, deadline):
    with (output / (name + ".log")).open("xb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("header audit wall allocation exhausted")
    if code:
        raise RuntimeError(f"{name} exited with code {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "identity-receipt", "facts-receipt", "engine-receipt", "facts", "enums", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if os.name != "posix" or args.output.is_relative_to(ROOT):
        raise ValueError("use Linux and a new private output directory outside the checkout")
    maps, facts, counts, inputs = expected_inputs(args)
    compiler = Path(shutil.which(args.compiler) or args.compiler).resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "audit.cpp"
    source.write_text(program(maps, facts))
    expected = {"identities": sum(len(pairs) for pairs in maps.values()),
                "unknowns": sum(len(unknown_ids({native for native, dense in pairs})) for pairs in maps.values()),
                "negative_skills": sum(native < 0 for native, dense in maps["skill"]), "facts": len(facts)}
    manifest = {"schema": "86-tcg-agent/native-policy-header-audit/1", "audit_source_sha256": sha(__file__),
                "input_artifacts": inputs, "compiler": str(compiler), "compiler_sha256": sha(compiler),
                "generated_source_sha256": sha(source), "expected": expected,
                "limits": {"wall_seconds": 180, "cpu_threads": 1, "nice": 19},
                "models_loaded": 0, "engine_libraries_loaded": 0, "native_games_started": 0}
    write(args.output / "manifest.json", manifest)
    os.nice(max(0, 19 - os.nice(0)))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    started = time.monotonic()
    deadline = started + 180
    try:
        command_run([str(compiler), "--version"], args.output, "compiler-version", deadline)
        binary = args.output / "audit"
        command_run([str(compiler), "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
                     "-I", str(args.facts_receipt.parent), str(source), "-o", str(binary)],
                    args.output, "compile", deadline)
        command_run([str(binary)], args.output, "execute", deadline)
        actual = json.loads((args.output / "execute.log").read_bytes())
        if actual != expected or any(sha(path) != expected_sha for path, expected_sha in inputs.items()):
            raise ValueError("compiled observations or pinned inputs differ")
        result = {"schema": manifest["schema"], "status": "passed", "actual": actual,
                  "native_categories": counts, "manifest_sha256": sha(args.output / "manifest.json"),
                  "binary_sha256": sha(binary), "wall_seconds": time.monotonic() - started,
                  "output_artifacts": {path.name: sha(path) for path in args.output.iterdir() if path.is_file()},
                  "zero_semantics": "ToDense(0) returns reserved dense0 and IsRegistered(0) is false. ForCard(0) returns Other/0 facts. Unknown nonzero card IDs return no optional facts.",
                  "scope": "Complete pinned catalog identity and static raw-fact parity through compiled C++20 lookups. No native engine, reader, placement, visibility, dynamic-prize or old-checkpoint compatibility proof.",
                  "models_loaded": 0, "engine_libraries_loaded": 0, "native_games_started": 0}
        write(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "actual": actual, "result_sha256": sha(args.output / "result.json")}))
    except Exception as error:
        write(args.output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
              "wall_seconds": time.monotonic() - started, "models_loaded": 0, "native_games_started": 0})
        raise


if __name__ == "__main__":
    main()
