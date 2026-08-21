import argparse
from pathlib import Path
import re

from utils.metrics import evaluate_dataset, write_result_files


DATASETS = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"]


def discover_latest_run(results_root: Path):
    candidates = []
    for path in results_root.iterdir() if results_root.is_dir() else []:
        match = re.match(r"^run(\d+)_", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path.name))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def discover_models(run_dir: Path):
    pattern = re.compile(r"^(?:last_model|best_model\d*)$", re.IGNORECASE)
    return [path.name for path in sorted(run_dir.iterdir()) if path.is_dir() and pattern.match(path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate static SAM2 predictions.")
    parser.add_argument("--data-path", type=str, default="data/TestDataset")
    parser.add_argument("--models", type=str, default=None, help="run directory name under results/")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    results_root = project_root / "results"
    run_name = args.models or discover_latest_run(results_root)
    if run_name is None:
        raise FileNotFoundError(f"No runN_* directory found in {results_root}")
    run_dir = results_root / run_name
    model_names = discover_models(run_dir)
    if not model_names:
        raise FileNotFoundError(f"No model result directories found in {run_dir}")
    data_root = Path(args.data_path)

    for model_name in model_names:
        model_result_dir = run_dir / model_name
        output_dir = project_root / "EvaluateResults" / run_name / model_name
        results = {}
        for dataset_name in args.datasets:
            prediction_dir = model_result_dir / dataset_name
            ground_truth_dir = data_root / dataset_name / "masks"
            if not prediction_dir.is_dir() or not ground_truth_dir.is_dir():
                print(f"[Skip] {dataset_name}: predictions or masks missing")
                continue
            result = evaluate_dataset(prediction_dir, ground_truth_dir)
            results[dataset_name] = result
            print(
                f"{run_name}/{model_name}/{dataset_name} | "
                f"meanDic={result['meanDic']:.4f} | meanIoU={result['meanIoU']:.4f}"
            )
        if results:
            write_result_files(output_dir, run_name, model_name, results)
            lines = []
            for dataset_name, result in results.items():
                lines.append(
                    f"Run={run_name} Model={model_name} Dataset={dataset_name} "
                    f"meanDic={result['meanDic']:.4f} meanIoU={result['meanIoU']:.4f} "
                    f"maxDic={result['maxDic']:.4f} maxIoU={result['maxIoU']:.4f}"
                )
            (output_dir / f"{model_name}_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
