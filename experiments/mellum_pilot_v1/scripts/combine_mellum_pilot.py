#!/usr/bin/env python3
"""Make a derived analysis index over completed, immutable dev/eval stage outputs.

Only the destination is written. Trace paths point back into the original stages;
keep dev/, eval/, and the derived pilot/ together when moving results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

VERSION = "mellum_combine_v1"
COMMON_METADATA = ("model", "environment", "generation_config_effective", "trace_spec", "semantics")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_info(path: Path, relative_to: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty required artifact: {path}")
    return {"path": os.path.relpath(path, relative_to), "bytes": path.stat().st_size, "sha256": sha256(path)}


def prepare(dataset: Path, stages: dict[str, Path], output: Path) -> tuple[list[dict], dict]:
    dataset, output = dataset.resolve(), output.resolve()
    stages = {split: directory.resolve() for split, directory in stages.items()}
    if set(stages) != {"dev", "eval"} or len(set(stages.values())) != 2:
        raise ValueError("Provide distinct dev and eval directories")
    for stage in stages.values():
        if output == stage or output in stage.parents or stage in output.parents:
            raise ValueError("Derived output must be separate from source stage directories")
    tasks = read_jsonl(dataset)
    by_id = {task["id"]: task for task in tasks}
    if not tasks or len(by_id) != len(tasks):
        raise ValueError("Empty dataset or duplicate base task IDs")
    if any(t["metadata"].get("split") not in stages for t in tasks):
        raise ValueError("Every task must have dev/eval metadata.split")
    if any(set(t["variants"]) != {"clean", "benign", "injection"} for t in tasks):
        raise ValueError("Each task must have clean, benign, and injection variants")

    # Split files from this frozen generator use the exact corresponding full-file lines.
    split_bytes = {split: b"" for split in stages}
    for line in dataset.read_bytes().splitlines(keepends=True):
        if line.strip():
            task = json.loads(line)
            split_bytes[task["metadata"]["split"]] += line
    combined, provenance, source_metadata = {}, {}, {}
    source_snapshots = {}
    for split, stage in stages.items():
        metadata_path = stage / "metadata.json"
        generation_path = stage / "generations.jsonl"
        metadata = read_json(metadata_path)
        if metadata.get("status") != "complete":
            raise ValueError(f"{split} stage is not complete")
        for field in COMMON_METADATA:
            if field not in metadata:
                raise ValueError(f"{split} stage is missing metadata.{field}")
        expected_tasks = {sid: t for sid, t in by_id.items() if t["metadata"]["split"] == split}
        expected = {(sid, condition) for sid, t in expected_tasks.items() for condition in t["variants"]}
        if not expected:
            raise ValueError(f"No {split} tasks in dataset")
        source_dataset = metadata.get("dataset", {})
        if source_dataset.get("sha256") != hashlib.sha256(split_bytes[split]).hexdigest():
            raise ValueError(f"{split} source dataset hash differs from frozen full-dataset split")
        if source_dataset.get("num_samples") != len(expected_tasks) or source_dataset.get("num_variants") != len(expected):
            raise ValueError(f"{split} source dataset counts differ from expected tasks/variants")
        source_metadata[split] = metadata
        source_files = {"metadata": file_info(metadata_path, output),
                        "generations": file_info(generation_path, output),
                        "config": file_info(stage / "config.yaml", output)}
        for info in source_files.values():
            source_snapshots[(output / info["path"]).resolve()] = info["sha256"]
        records = read_jsonl(generation_path)
        seen = set()
        trace_files = []
        for record in records:
            key = (record["sample_id"], record["condition"])
            if key in seen or key in combined:
                raise ValueError(f"Duplicate generation identity: {key}")
            if key not in expected:
                raise ValueError(f"Unexpected {split} task/condition: {key}")
            seen.add(key)
            task = by_id[key[0]]
            if record.get("metadata") != task["metadata"] or record.get("evaluation") != task["evaluation"]:
                raise ValueError(f"Task provenance/answer differs: {key}")
            if record.get("messages") != task["variants"][key[1]]["messages"]:
                raise ValueError(f"Generation messages differ from dataset: {key}")
            if not isinstance(record.get("response_answer"), str) or record.get("finish_reason") not in {"eos", "length"}:
                raise ValueError(f"Missing generation outcome: {key}")
            source_trace = record.get("trace_dir")
            if not isinstance(source_trace, str) or Path(source_trace).is_absolute():
                raise ValueError(f"Expected relative source trace path: {key}")
            trace = (stage / source_trace).resolve()
            if stage not in trace.parents:
                raise ValueError(f"Trace escapes source stage: {key}")
            trace_meta = read_json(trace / "trace.json")
            if (trace_meta.get("sample_id"), trace_meta.get("condition")) != key:
                raise ValueError(f"Trace identity differs: {key}")
            if trace_meta.get("input_ids") != record.get("input_ids"):
                raise ValueError(f"Trace and generation token IDs differ: {key}")
            if not isinstance(trace_meta.get("routing", {}).get("num_experts"), int):
                raise ValueError(f"Trace has no expert-count metadata: {key}")
            required = ["trace.json", "routing.safetensors"]
            required.extend(f"{kind}.safetensors" for kind in ("activations", "attention") if kind in trace_meta)
            for name in required:
                info = file_info(trace / name, output)
                trace_files.append(info)
                source_snapshots[(trace / name).resolve()] = info["sha256"]
            derived = dict(record)
            derived["trace_dir"] = os.path.relpath(trace, output)
            combined[key] = derived
        if seen != expected:
            raise ValueError(f"{split} stage is missing {len(expected - seen)} expected variants")
        provenance[split] = {"directory": os.path.relpath(stage, output), "source_files": source_files,
                             "source_metadata": metadata, "trace_files": trace_files,
                             "n_tasks": len(expected_tasks), "n_variants": len(records)}

    for field in COMMON_METADATA:
        if source_metadata["dev"][field] != source_metadata["eval"][field]:
            raise ValueError(f"Cannot combine stages with different metadata.{field}")
    # Detect files changing while we read/hash them; never index an active run.
    for path, expected_hash in source_snapshots.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"Source changed while combining: {path}")
    rows = [combined[(t["id"], condition)] for t in tasks for condition in t["variants"]]
    metadata = {"artifact_kind": "derived_stage_collection", "created_by": "scripts/combine_mellum_pilot.py",
                "combiner_version": VERSION, "combiner_sha256": sha256(Path(__file__)),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "experiment_name": "derived_mellum_pilot", "status": "derived_complete",
                "dataset": {"path": os.path.relpath(dataset, output), "sha256": sha256(dataset),
                            "num_samples": len(tasks), "num_variants": len(rows)},
                "sources": provenance,
                "common_metadata_provenance": "Fields below are copied from source metadata after equality validation across dev/eval.",
                **{field: source_metadata["dev"][field] for field in COMMON_METADATA},
                "source_run_time_bounds": {"started_at": min(m["started_at"] for m in source_metadata.values()),
                                           "finished_at": max(m["finished_at"] for m in source_metadata.values())},
                "derivation": {"only_generation_record_change": "trace_dir rebased relative to derived output",
                               "ordering": "full dataset task order, then variant order",
                               "traces": "Referenced in original stages; no model rerun, tensor modification, or copied traces.",
                               "analysis": {"reference": "clean", "control": "benign", "alignment": "span", "span": "question"}},
                "validation": {"exact_expected_ids_conditions": True, "source_dataset_hashes": True,
                               "source_prompts_answers_provenance": True, "trace_identity_tokens": True,
                               "required_trace_files_exist_and_hashed": True,
                               "tensor_contents_loaded": False}}
    return rows, metadata


def combine(dataset: Path, dev_dir: Path, eval_dir: Path, output: Path, check_only: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and not check_only:
        raise FileExistsError(f"Refusing to overwrite derived output: {output}")
    rows, metadata = prepare(dataset, {"dev": dev_dir, "eval": eval_dir}, output)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    metadata["derived_generations_sha256"] = hashlib.sha256(data).hexdigest()
    if not check_only:
        output.mkdir(parents=True, exist_ok=False)
        (output / "generations.jsonl").write_bytes(data)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (output / "README.md").write_text(
            "# Derived Mellum pilot index\n\n"
            "This directory combines completed dev/eval records for analysis; it is not an upstream model run. "
            "Original records and traces are unchanged. Relative trace references require both source stage directories "
            "to remain available. Source paths, metadata snapshots, hashes, and checks are in metadata.json. "
            "No combined runner config is invented.\n\n"
            "Run upstream CPU analysis with explicit `--alignment span --span question --reference clean --control benign`. "
            "Run the strict scorer with the full frozen examples.jsonl and this directory's generations.jsonl. "
            "Separate development and evaluation tasks in reporting; combining is only a storage/analysis convenience.\n",
            encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Frozen full examples.jsonl")
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New derived pilot directory")
    parser.add_argument("--check-only", action="store_true", help="Validate sources without writing")
    args = parser.parse_args()
    metadata = combine(args.dataset, args.dev_dir, args.eval_dir, args.output_dir, args.check_only)
    print(json.dumps({"status": "checked" if args.check_only else "derived_complete",
                      "tasks": metadata["dataset"]["num_samples"], "variants": metadata["dataset"]["num_variants"],
                      "dataset_sha256": metadata["dataset"]["sha256"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
