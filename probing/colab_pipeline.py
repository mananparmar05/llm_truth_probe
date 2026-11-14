"""Extract checkpointed activations and evaluate probes on held-out questions."""

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import h5py
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.probe_evaluator import ProbeEvaluator
from probing.probe_trainer import PROBE_FACTORIES

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("truthfulqa", "halueval")
SPLITS = ("train", "val", "test")
SEED = 42


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_rows(dataset, limit):
    paths = [ROOT / "data" / dataset / f"{split}.csv" for split in SPLITS]
    rows = pd.concat([pd.read_csv(path, keep_default_na=False) for path in paths],
                     ignore_index=True)
    questions = rows.question.str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
    rng = np.random.default_rng(SEED)
    if limit:
        chosen = rng.choice(questions.unique(), size=min(limit, questions.nunique()), replace=False)
        keep = questions.isin(chosen)
        rows = rows.loc[keep].reset_index(drop=True)
        questions = questions.loc[keep].reset_index(drop=True)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train, holdout = next(split.split(rows, groups=questions))
    half = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val, test = next(half.split(holdout, groups=questions.iloc[holdout]))
    partitions = dict(zip(SPLITS, (train, holdout[val], holdout[test])))
    for name, indices in partitions.items():
        if set(rows.label.iloc[indices]) != {0, 1}:
            raise ValueError(f"{dataset}/{name} must contain both labels. Increase --limit.")
        rows.loc[indices, "split"] = name
    return rows


def shard_paths(output, manifest, dataset):
    step = manifest["chunk_rows"]
    for start in range(0, manifest["row_counts"][dataset], step):
        end = min(manifest["row_counts"][dataset], start + step)
        yield start, end, output / "activations" / f"{dataset}-{start:06d}.h5"


def valid_shard(path, expected_shape):
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        if handle["states"].shape != expected_shape:
            raise ValueError(f"Unexpected activation shape in {path}")
    return True


def extract(args):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from extraction.batch_runner import BatchRunner
    from extraction.hook_extractor import HiddenStateExtractor

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before extracting activations.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "activations").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    revision = args.revision
    if previous.get("model") == args.model and previous.get("requested_revision") == revision:
        commit = previous["revision"]
    else:
        commit = HfApi().model_info(args.model, revision=revision).sha
    config = AutoConfig.from_pretrained(args.model, revision=commit)
    layers = sorted(set(args.layers or range(config.num_hidden_layers)))
    if min(layers) < 0 or max(layers) >= config.num_hidden_layers:
        raise ValueError(f"Invalid layer indices for a {config.num_hidden_layers}-layer model.")
    tables = {name: prepare_rows(name, args.limit) for name in DATASETS}
    row_csv = {name: table.to_csv(index=False) for name, table in tables.items()}
    manifest = {
        "model": args.model,
        "requested_revision": revision,
        "revision": commit,
        "quantization": "nf4",
        "layers": layers,
        "hidden_dim": config.hidden_size,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "chunk_rows": args.chunk_rows,
        "limit_questions_per_dataset": args.limit,
        "seed": SEED,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "row_counts": {name: len(table) for name, table in tables.items()},
        "row_sha256": {name: hashlib.sha256(csv.encode()).hexdigest() for name, csv in row_csv.items()},
    }
    if previous and previous != manifest:
        raise ValueError("This Drive folder belongs to another run. Select a new --output folder.")
    for name, csv in row_csv.items():
        (output / f"{name}_rows.csv").write_text(csv, encoding="utf-8", newline="")
    write_json(manifest_path, manifest)
    pending = []
    for name in DATASETS:
        for start, end, path in shard_paths(output, manifest, name):
            if not valid_shard(path, (end - start, len(layers), config.hidden_size)):
                pending.append((name, start, end, path))
    if not pending:
        print("All activation shards already saved.", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=commit, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=commit,
        device_map={"": 0},
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.config.use_cache = False
    extractor = HiddenStateExtractor(model, layer_indices=layers)
    # Run the decoder directly because the probes do not use vocabulary logits.
    runner = BatchRunner(model.base_model, tokenizer, extractor,
                         batch_size=args.batch_size, max_length=args.max_length)
    for number, (name, start, end, path) in enumerate(pending, 1):
        print(f"Shard {number}/{len(pending)} | {name} rows {start} to {end - 1}", flush=True)
        rows = tables[name].iloc[start:end]
        states, _ = runner.run(list(zip(rows.text, rows.label)), dataset_name=name)
        if not np.isfinite(states).all():
            raise ValueError(f"Non-finite activations in {name} rows {start}:{end}")
        temporary = path.with_suffix(".tmp")
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("states", data=states.astype(np.float16),
                                  chunks=(min(128, len(states)), 1, states.shape[2]),
                                  compression="lzf")
        temporary.replace(path)


def load_layer(output, manifest, dataset, column):
    rows = manifest["row_counts"][dataset]
    features = np.empty((rows, manifest["hidden_dim"]), dtype=np.float32)
    for start, end, path in shard_paths(output, manifest, dataset):
        with h5py.File(path, "r") as handle:
            features[start:end] = handle["states"][:, column, :]
    return features


def metrics(probe, features, rows, split):
    mask = rows.split == split
    return ProbeEvaluator(probe, features[mask], rows.label[mask]).all_metrics()


def train(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    tables = {}
    for name in DATASETS:
        path = output / f"{name}_rows.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["row_sha256"][name]:
            raise ValueError(f"Row manifest does not match saved activations: {path}")
        tables[name] = pd.read_csv(path, keep_default_na=False)
        for start, end, shard in shard_paths(output, manifest, name):
            if not valid_shard(shard, (end - start, len(manifest["layers"]), manifest["hidden_dim"])):
                raise FileNotFoundError(f"Missing {shard}. Resume the extraction cell.")

    results_dir = output / args.result_name
    results_dir.mkdir(exist_ok=True)
    training_path = results_dir / "training.json"
    training = {"probe_types": args.probes, "probe_seed": SEED, "svm": "calibrated LinearSVC"}
    if training_path.exists() and json.loads(training_path.read_text()) != training:
        raise ValueError("Probe settings changed. Use a new --result-name.")
    write_json(training_path, training)
    all_validation = []
    transfer = []
    for column, layer in enumerate(manifest["layers"]):
        checkpoint = results_dir / f"layer_{layer:02d}.json"
        if checkpoint.exists():
            layer_result = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            print(f"Training probes at layer {layer}", flush=True)
            features = {name: load_layer(output, manifest, name, column) for name in DATASETS}
            layer_result = {"validation": [], "transfer": []}
            for name, rows in tables.items():
                mask = rows.split == "train"
                for kind in args.probes:
                    probe = PROBE_FACTORIES[kind]()
                    probe.fit(features[name][mask], rows.label[mask])
                    joblib.dump(probe, results_dir / f"{name}_layer{layer:02d}_{kind}.pkl")
                    layer_result["validation"].append({
                        "dataset": name, "layer": layer, "probe": kind,
                        **metrics(probe, features[name], rows, "val"),
                    })
                    if kind == "logistic_regression":
                        for target, target_rows in tables.items():
                            layer_result["transfer"].append({
                                "train_dataset": name, "test_dataset": target, "layer": layer,
                                **metrics(probe, features[target], target_rows, "test"),
                            })
            write_json(checkpoint, layer_result)
        all_validation.extend(layer_result["validation"])
        transfer.extend(layer_result["transfer"])

    validation = pd.DataFrame(all_validation)
    validation.to_csv(results_dir / "validation_metrics.csv", index=False)
    transfer_table = pd.DataFrame(transfer)
    transfer_table.to_csv(results_dir / "cross_dataset_metrics.csv", index=False)
    selected = []
    for name, rows in tables.items():
        best = validation.loc[validation.dataset.eq(name)].sort_values(
            ["auroc", "layer", "probe"], ascending=[False, True, True]
        ).iloc[0]
        layer = int(best.layer)
        filename = f"{name}_layer{layer:02d}_{best.probe}.pkl"
        probe = joblib.load(results_dir / filename)
        features = load_layer(output, manifest, name, manifest["layers"].index(layer))
        mask = rows.split == "test"
        evaluator = ProbeEvaluator(probe, features[mask], rows.label[mask])
        evaluator.plot_roc_curve(save_path=results_dir / f"{name}_roc.png")
        plt.close()
        evaluator.plot_confusion_matrix(save_path=results_dir / f"{name}_confusion.png")
        plt.close()
        selected.append({
            "dataset": name, "layer": layer, "probe": best.probe, "checkpoint": filename,
            "validation_auroc": float(best.auroc), **evaluator.all_metrics(),
        })
        ax = validation.loc[validation.dataset.eq(name)].pivot(
            index="layer", columns="probe", values="auroc"
        ).plot(marker=".", ylabel="Validation AUROC", title=name)
        ax.figure.savefig(results_dir / f"{name}_layers.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    ax = transfer_table.pivot(
        index="layer", columns=["train_dataset", "test_dataset"], values="auroc"
    ).plot(marker=".", ylabel="Test AUROC", title="Logistic probe transfer")
    ax.figure.savefig(results_dir / "cross_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    pd.DataFrame(selected).to_csv(results_dir / "test_metrics.csv", index=False)
    print(pd.DataFrame(selected).to_string(index=False), flush=True)
    print(f"Results saved to {results_dir}", flush=True)


def run_controls(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results_dir = output / args.result_name
    validation = pd.read_csv(results_dir / "validation_metrics.csv")
    records = []
    for dataset in DATASETS:
        path = output / f"{dataset}_rows.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["row_sha256"][dataset]:
            raise ValueError(f"Row manifest does not match saved activations: {path}")
        rows = pd.read_csv(path, keep_default_na=False)
        candidates = validation.loc[
            validation.dataset.eq(dataset) & validation.probe.eq("logistic_regression")
        ]
        best = candidates.sort_values(["auroc", "layer"], ascending=[False, True]).iloc[0]
        layer = int(best.layer)
        features = load_layer(output, manifest, dataset, manifest["layers"].index(layer))
        mask = rows.split == "train"
        labels = rows.label[mask].to_numpy()
        probe = joblib.load(results_dir / f"{dataset}_layer{layer:02d}_logistic_regression.pkl")
        records.append({
            "dataset": dataset, "layer": layer, "control": "activation_probe", "seed": SEED,
            **metrics(probe, features, rows, "test"),
        })
        for seed in range(SEED, SEED + args.repeats):
            shuffled = np.random.default_rng(seed).permutation(labels)
            probe = PROBE_FACTORIES["logistic_regression"]()
            probe.fit(features[mask], shuffled)
            records.append({
                "dataset": dataset, "layer": layer, "control": "shuffled_labels", "seed": seed,
                **metrics(probe, features, rows, "test"),
            })
        lengths = np.column_stack([
            rows.question.str.len(), rows.answer.str.len(),
            rows.question.str.split().str.len(), rows.answer.str.split().str.len(),
        ])
        probe = PROBE_FACTORIES["logistic_regression"]()
        probe.fit(lengths[mask], labels)
        records.append({
            "dataset": dataset, "layer": None, "control": "text_length", "seed": SEED,
            **metrics(probe, lengths, rows, "test"),
        })

    table = pd.DataFrame(records)
    table["layer"] = table["layer"].astype("Int64")
    table.to_csv(results_dir / "control_metrics.csv", index=False)
    summary = table.groupby(["dataset", "control"]).auroc.agg(["mean", "std", "count"])
    ax = summary["mean"].unstack().plot.bar(rot=0, ylabel="Held-out AUROC", ylim=(0, 1))
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.figure.savefig(results_dir / "control_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    evaluation = commands.add_parser("train")
    controls = commands.add_parser("controls")
    for command in (extraction, evaluation, controls):
        command.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--model", default="Qwen/Qwen2.5-7B")
    extraction.add_argument("--revision", default="main")
    extraction.add_argument("--layers", type=int, nargs="+")
    extraction.add_argument("--batch-size", type=int, default=4)
    extraction.add_argument("--max-length", type=int, default=512)
    extraction.add_argument("--chunk-rows", type=int, default=512)
    extraction.add_argument("--limit", type=int, default=0, help="Questions per dataset; 0 uses all.")
    evaluation.add_argument("--probes", nargs="+", choices=PROBE_FACTORIES,
                            default=["logistic_regression", "svm"])
    evaluation.add_argument("--result-name", default="results")
    controls.add_argument("--result-name", default="results")
    controls.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "extract":
        if min(args.batch_size, args.max_length, args.chunk_rows) < 1 or args.limit < 0:
            parser.error("Batch size, max length and chunk rows must be positive; limit must be nonnegative.")
        extract(args)
    elif args.command == "train":
        if "logistic_regression" not in args.probes or len(set(args.probes)) != len(args.probes):
            parser.error("Provide unique probes including logistic_regression for the transfer study.")
        with threadpool_limits(limits=2):
            train(args)
    else:
        if args.repeats < 1:
            parser.error("--repeats must be positive.")
        with threadpool_limits(limits=2):
            run_controls(args)


if __name__ == "__main__":
    main()
