"""Score the ASVspoof2019 LA evaluation partition with a trained checkpoint and
write the submission csv.

Usage:
    python3 make_submission.py \
        --checkpoint saved/<run_name>/model_best.pth \
        --out <your_university_login>.csv

The csv format is dictated by grading.py: exactly two comma-separated columns,
trial key and score, and NO HEADER ROW (a header row would be parsed as data and
crash the float() conversion). Every key of the evaluation protocol must be
present, otherwise grading.py raises KeyError.

Use --part val first: the script prints the pooled EER of the partition, which
must match the value logged by the trainer for the same checkpoint. A mismatch
means the inference-time preprocessing differs from the one used in training.

--legacy-normalize exists for checkpoints trained with the template's example
batch transform (`Normalize1D(mean=0.5, std=0.5)`, i.e. x -> 2x - 1) still
enabled in the config. That transform is applied by the trainer but not by this
script, so without the flag such a checkpoint would be scored on inputs it has
never seen. Models trained with `transforms: asvspoof` do not need the flag.
"""

import argparse
import csv

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.metrics.eer import compute_eer_numpy, scores_from_logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="saved/fft-lcnn/model_best.pth",
        help="path to the trained checkpoint",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output csv; must be named <your_university_login>.csv",
    )
    parser.add_argument("--part", default="test", help="dataset key: test or val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override data_dir stored in the checkpoint config",
    )
    parser.add_argument(
        "--legacy-normalize",
        action="store_true",
        help=(
            "apply (x - 0.5) / 0.5 to the waveform, reproducing the example "
            "batch transform; needed only for checkpoints trained with it"
        ),
    )
    return parser.parse_args()


def write_csv(path, file_ids, scores):
    # no header: grading.py treats every 2-column row as data
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for file_id, score in zip(file_ids, scores):
            writer.writerow([file_id, f"{score:.6f}"])


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    if args.data_dir is not None:
        OmegaConf.update(config, "data_dir", args.data_dir, force_add=True)

    model = instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    if args.part not in config.datasets:
        raise KeyError(
            f"partition '{args.part}' is not in the config; "
            f"available: {list(config.datasets.keys())}"
        )

    dataset = instantiate(config.datasets[args.part])
    n_trials_expected = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if args.legacy_normalize:
        print("Applying legacy normalisation (x - 0.5) / 0.5 to the waveform")

    all_scores = []
    all_labels = []
    all_file_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=args.part):
            data_object = batch["data_object"].to(device)

            if args.legacy_normalize:
                data_object = (data_object - 0.5) / 0.5

            logits = model(data_object=data_object)["logits"]

            all_scores.append(scores_from_logits(logits).cpu())
            all_labels.append(batch["labels"].long())
            all_file_ids.extend(batch["file_id"])

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    # sanity checks mirroring what grading.py will do
    assert len(all_file_ids) == n_trials_expected, (
        f"scored {len(all_file_ids)} trials but the protocol lists "
        f"{n_trials_expected}"
    )
    assert len(set(all_file_ids)) == len(all_file_ids), "duplicate trial keys"
    assert not (scores != scores).any(), "scores contain NaN"
    n_unique = len(set(scores.tolist()))
    assert n_unique >= 3, (
        f"only {n_unique} distinct score values: grading.py rejects hard "
        "decisions, soft scores are required"
    )

    print(f"\nScored trials: {len(scores)}")

    bonafide = scores[labels == 1]
    spoof = scores[labels == 0]
    if len(bonafide) > 0 and len(spoof) > 0:
        eer, threshold = compute_eer_numpy(bonafide, spoof)
        print(f"Bona fide: {len(bonafide)}, spoof: {len(spoof)}")
        print(f"Pooled EER: {eer:.4f}%  (threshold {threshold:.4f})")
        print(
            "Compare this with the EER logged by the trainer for the same "
            "checkpoint and partition; the two must agree."
        )
    else:
        print("Labels unavailable for this partition, EER not computed.")

    write_csv(args.out, all_file_ids, scores)
    print(f"Wrote {args.out} ({len(all_file_ids)} rows, no header)")


if __name__ == "__main__":
    main()
