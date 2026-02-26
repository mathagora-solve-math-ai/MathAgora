#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_runner_module() -> Any:
    runner_path = Path(__file__).resolve().with_name("run_backend_solve_parallel_5y.py")
    spec = importlib.util.spec_from_file_location("runner_mod", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load runner module: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolution path expects module to be present in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Retry exactly one record via OpenRouter model=openai/gpt-5-codex "
            "(no temperature, no max_tokens parameter) and append to existing outputs."
        )
    )
    p.add_argument("--year", required=True, help="Target year (e.g. 2024)")
    p.add_argument("--prob-id", required=True, help="Target prob_id (e.g. 2024_odd_geometry_28)")
    p.add_argument("--run-index", type=int, required=True, help="Target run index (1..N)")
    p.add_argument("--output-dir", default="0212/poc/output")
    p.add_argument("--output-prefix", default="backend_solve_parallel_5y")
    p.add_argument("--max-attempts", type=int, default=2, help="Retry attempts (default: 2)")
    p.add_argument(
        "--replace-existing-key",
        action="store_true",
        help=(
            "If set, remove existing rows with same (year, prob_id, model_id, run_index) "
            "from flat/raw before appending the new result."
        ),
    )
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def backup_if_exists(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    backup.write_bytes(path.read_bytes())
    return backup


def drop_existing_key_from_outputs(
    runner: Any,
    *,
    flat_csv_path: Path,
    raw_jsonl_path: Path,
    key: tuple[str, str, str, int],
) -> tuple[int, int]:
    target_year, target_prob_id, target_model_id, target_run_idx = key
    removed_flat = 0
    removed_raw = 0

    if flat_csv_path.exists():
        with flat_csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        kept: list[dict[str, Any]] = []
        for r in rows:
            k = (
                str(r.get("year", "")).strip(),
                str(r.get("prob_id", "")).strip(),
                str(r.get("model_id", "")).strip(),
                int(str(r.get("run_index", "0")).strip() or "0"),
            )
            if k == key:
                removed_flat += 1
            else:
                kept.append(r)
        with flat_csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=runner.FLAT_HEADER)
            w.writeheader()
            for r in kept:
                w.writerow({h: r.get(h, "") for h in runner.FLAT_HEADER})

    if raw_jsonl_path.exists():
        kept_lines: list[str] = []
        with raw_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    o = json.loads(s)
                except Exception:
                    kept_lines.append(line)
                    continue
                k = (
                    str(o.get("year", "")).strip(),
                    str(o.get("prob_id", "")).strip(),
                    str(o.get("model_id", "")).strip(),
                    int(str(o.get("run_index", "0")).strip() or "0"),
                )
                if k == key:
                    removed_raw += 1
                else:
                    kept_lines.append(line)
        with raw_jsonl_path.open("w", encoding="utf-8") as f:
            for ln in kept_lines:
                f.write(ln)

    return removed_flat, removed_raw


def main() -> None:
    args = parse_args()
    runner = load_runner_module()

    year = str(args.year).strip()
    prob_id = str(args.prob_id).strip()
    run_index = int(args.run_index)
    if run_index <= 0:
        raise ValueError("--run-index must be >= 1")

    out_dir = Path(args.output_dir)
    flat_csv_path = out_dir / f"{args.output_prefix}_flat.csv"
    raw_jsonl_path = out_dir / f"{args.output_prefix}_raw.jsonl"
    summary_json_path = out_dir / f"{args.output_prefix}_summary.json"

    model_id = "openai/gpt-5-codex"
    model_key = "GPT-5-Codex(OpenRouter)"
    target_key = (year, prob_id, model_id, run_index)

    problems = runner.load_problems_from_year_tsv(year, 0)
    target = next((p for p in problems if p.prob_id == prob_id), None)
    if target is None:
        raise RuntimeError(f"Problem not found: year={year}, prob_id={prob_id}")

    print("=== Retry single key via OpenRouter codex ===")
    print(f"Target: year={year}, prob_id={prob_id}, model_id={model_id}, run_index={run_index}")
    print(f"Output dir: {out_dir}")
    print(f"Replace existing key: {bool(args.replace_existing_key)}")
    print(f"Max attempts: {int(args.max_attempts)}")
    print("Token behavior: max_tokens NOT sent")
    print("Temperature behavior: temperature NOT sent")

    ensure_parent(flat_csv_path)
    ensure_parent(raw_jsonl_path)
    ensure_parent(summary_json_path)

    if args.replace_existing_key:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        flat_bak = backup_if_exists(flat_csv_path, stamp)
        raw_bak = backup_if_exists(raw_jsonl_path, stamp)
        removed_flat, removed_raw = drop_existing_key_from_outputs(
            runner,
            flat_csv_path=flat_csv_path,
            raw_jsonl_path=raw_jsonl_path,
            key=target_key,
        )
        print(f"Removed existing key rows: flat={removed_flat}, raw={removed_raw}")
        if flat_bak:
            print(f"Flat backup: {flat_bak}")
        if raw_bak:
            print(f"Raw backup: {raw_bak}")

    messages = runner.build_user_messages(target)
    chat_messages = runner.make_chat_messages_with_system(runner.V4_PROMPT_SIMPLE, messages)
    schema = runner.schema_for_model(model_id)
    client = runner.get_openrouter_client()

    response_text = ""
    error_message = ""
    attempts = max(1, int(args.max_attempts))
    elapsed_last = 0.0
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            response_text = runner.call_chat_json_schema(
                client=client,
                model=model_id,
                messages=chat_messages,
                schema=schema,
                max_tokens_field="max_tokens",
                max_tokens=None,  # no max token parameter
                temperature=None,  # no temperature parameter
            )
            elapsed_last = time.time() - t0
            break
        except Exception as exc:
            elapsed_last = time.time() - t0
            error_message = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(2.0 * (2 ** (attempt - 1)))

    parse_ok = False
    step_count = 0
    titles: list[str] = []
    final_answer = ""
    if response_text:
        parsed = runner.parse_model_solution_json(response_text)
        if parsed is not None:
            step_count = int(parsed["step_count"])
            titles = list(parsed["titles"])
            final_answer = str(parsed["final_answer"]).strip()
            parse_ok = bool(step_count > 0 and final_answer)

    gold_answer = str(target.answer).strip()
    gold_choice = runner.extract_choice_label(gold_answer)
    model_choice = runner.extract_choice_label(final_answer)
    correct = runner.is_correct_answer(final_answer, gold_answer) if final_answer else False
    correct_recheck = runner.is_correct_answer_recheck(final_answer, gold_answer) if final_answer else False

    row: dict[str, Any] = {
        "recorded_at_utc": runner.utc_now_iso(),
        "year": target.year,
        "source_file": target.source_file,
        "prob_id": target.prob_id,
        "prob_type": target.prob_type,
        "model": model_key,
        "model_id": model_id,
        "run_index": run_index,
        "temperature_sent": "",
        "step_count": step_count,
        "parse_ok": parse_ok,
        "model_final_answer": final_answer,
        "gold_answer": gold_answer,
        "gold_choice": gold_choice,
        "model_choice": model_choice,
        "correct": correct,
        "correct_recheck": correct_recheck,
        "elapsed_time_seconds": round(elapsed_last, 6),
        "error": error_message,
        "titles": titles,
        "response_text": response_text,
    }

    runner.append_flat_row(flat_csv_path, row)
    runner.append_raw_record(raw_jsonl_path, row)

    summary = runner.summarize_flat(flat_csv_path)
    existing_meta: dict[str, Any] = {}
    if summary_json_path.exists():
        try:
            existing_meta = (json.loads(summary_json_path.read_text(encoding="utf-8")).get("meta") or {})
        except Exception:
            existing_meta = {}
    existing_meta.update(
        {
            "last_single_retry_at_utc": runner.utc_now_iso(),
            "last_single_retry_target": {
                "year": year,
                "prob_id": prob_id,
                "model_id": model_id,
                "run_index": run_index,
                "route": "openrouter",
                "temperature_sent": "",
                "max_tokens_sent": "",
                "max_tokens_policy": "unset",
            },
        }
    )
    summary["meta"] = existing_meta
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Done ===")
    print(f"Elapsed(last attempt): {elapsed_last:.3f}s")
    print(f"Parse OK: {parse_ok}")
    print(f"Correct: {correct} / Correct(recheck): {correct_recheck}")
    print(f"Error: {error_message or '(none)'}")
    print(f"Flat CSV: {flat_csv_path}")
    print(f"Raw JSONL: {raw_jsonl_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
