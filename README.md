# VeriSketch research software

Research code for federated text classification, masked aggregation, sketch-space filtering,
DP-SGD experiments and aggregate evaluation. This snapshot contains code and aggregate
results. It does not include the unpublished manuscript, manuscript-generation sources,
review correspondence, experiment logs, individual predictions or trained models.

The code is a research prototype. Included measurements are an experimental snapshot;
their presence does not certify deployment security or differential privacy.

## Contents

- `secagg.py`, `fl.py`: protocol primitives, models, training and aggregation.
- `data_prep.py`, `features.py`: local data preparation and feature extraction.
- `exp.py` and the evaluation scripts: experiment execution on locally supplied inputs.
- `config/defaults.json`: non-secret example training configuration.
- `public_results/`: aggregate run metrics, confusion matrices and aggregate evaluation results.
- `analyze.py`: validate confusion-matrix metrics and summarize released runs using Python's standard library.
- `release_results.py`, `release_check.py`: explicit aggregate export and content checks.

## Inspect aggregate results

No training dependencies or datasets are needed for these commands:

```bash
python release_check.py
python analyze.py --input public_results --output results/aggregate_summary.csv
```

In moderation results, `harm05` denotes harm probability above 0.5; it is not multiclass argmax.
The summaries preserve aggregate measurements; they do not contain record-level predictions,
post identifiers, client class distributions or per-client inference errors. Post-level paired
attribution tests cannot be independently reconstructed from aggregate summaries alone.

## Run experiments locally

Use Python 3.13 and install `requirements.txt` in a virtual environment. Supply these inputs
under their original licenses and terms; they are not redistributed here:

- `data/cyberbullying_tweets.csv`: Kaggle Cyberbullying Classification.
- `data/dataset.json`, `data/post_id_divisions.json`: HateXplain and its official split.
- `model/`: the DistilBERT-base-uncased weights, config and tokenizer files expected by `features.py`.

```bash
python data_prep.py
python features.py
python test_secagg.py
python test_dp.py
python smoke_test.py
python exp.py baseline
```

Other experiment groups are `robust`, `joint`, `dp`, `nullspace`, `sketchdim`, `fsweep` and `elsa`.
The `elsa` group emulates norm-bound filtering; it is not an implementation of ELSA cryptography.
Training and evaluations write into local `results/`; input data, features, models and all
new experiment outputs are excluded by `.gitignore`. Secure masks and experimental DP
random streams prevent exact bit-for-bit repetition of all runs.

## Prepare another release

Never copy the local working directory wholesale. Export into a new empty staging directory:

```bash
python release_results.py --source results --destination release_staging/public_results --manifest data/kaggle_clean_v2.csv
```

This removes record/client fields and derives aggregate confusion matrices when a local
manifest is supplied. It is a content-minimization step, not a DP mechanism or a guarantee
of anonymity. Run the content check on the complete clean staging snapshot before upload.
The ignore file does not remove files that were previously tracked or published.

## License and citation

Code and released aggregate results use the MIT License. Author attribution is in `LICENSE`
and `CITATION.cff`. No publication DOI, journal acceptance or public repository address is claimed.
