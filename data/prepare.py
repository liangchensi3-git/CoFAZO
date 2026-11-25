"""Rebuild the seven paper datasets from pinned public releases."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
TREC_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")


def source_file(spec, item, raw_root):
    if raw_root is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
            filename=item["file"], token=False,
        ))
    else:
        path = raw_root / spec["repo"].replace("/", "__") / item["file"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"Source checksum mismatch for {item['file']}")
    return path


def source_rows(path, filename):
    if filename.endswith(".jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
    else:
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()


def canonical(task, row, index, source_split):
    if task in ("sst2", "rte"):
        return row
    label = (TREC_LABELS.index(row["label_coarse_original"])
             if task == "trec" else int(row["label"]))
    if task == "mnli":
        text, pair = row["premise"], row["hypothesis"]
    else:
        text, pair = row["sentence" if task == "cola" else "text"], ""
    return dict(idx=index, text=text, text_pair=pair, label=label, eval_split=source_split)


def input_key(row):
    return tuple(" ".join(row[field].lower().split()) for field in ("text", "text_pair"))


def build(task, raw_root):
    from datasets import Dataset, DatasetDict

    spec = SOURCES[task]
    rows = {"train": [], "validation": []}
    for item in spec["files"]:
        path = source_file(spec, item, raw_root)
        split = item["split"]
        target = rows["train" if split == "train" else "validation"]
        for row in source_rows(path, item["file"]):
            target.append(canonical(task, row, len(target), split))
    if task in ("cola", "trec"):
        held_out = {input_key(row) for row in rows["validation"]}
        rows["train"] = [row for row in rows["train"] if input_key(row) not in held_out]
    return DatasetDict({name: Dataset.from_list(records) for name, records in rows.items()})


def record_hash(dataset):
    # Hash canonical JSON records in on-disk order, independently of Arrow metadata.
    digest = hashlib.sha256(b"[")
    for start in range(0, len(dataset), 2048):
        columns = dataset[start:start + 2048]
        records = [dict(zip(columns, values)) for values in zip(*columns.values())]
        block = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if start:
            digest.update(b",")
        digest.update(block[1:-1].encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def verify(task, directory):
    from datasets import load_from_disk

    data = load_from_disk(str(directory))
    for name, expected in SOURCES[task]["splits"].items():
        split = data[name]
        if len(split) != expected["count"] or record_hash(split) != expected["records_sha256"]:
            raise ValueError(f"{task}/{name}: records differ from the paper snapshot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *SOURCES], default="all")
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    parser.add_argument("--raw-root", type=Path, help="Offline mirror, laid out as described in README.md")
    parser.add_argument("--verify-only", action="store_true", help="Check existing splits without downloading")
    args = parser.parse_args()
    for task in (SOURCES if args.task == "all" else [args.task]):
        destination = args.output / task
        if destination.exists() or args.verify_only:
            verify(task, destination)
        else:
            data = build(task, args.raw_root)
            args.output.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=f".{task}-", dir=args.output) as tmp:
                staged = Path(tmp) / task
                data.save_to_disk(str(staged))
                verify(task, staged)
                staged.rename(destination)
        print(f"Verified {task} at {destination}", flush=True)


if __name__ == "__main__":
    main()
