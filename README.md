# VeriSketch

Research code for VeriSketch, a two-server secure aggregation protocol in which the aggregation
server filters clients on owner-checked count sketches of their updates and learns only one
aggregate per round. The code covers the protocol over the prime field F_P (P = 2^61 - 1),
federated training of a classification head on frozen DistilBERT features, DP-SGD, Byzantine
attacks, plaintext robust baselines, and the evaluation studies (leakage, integrity, cost,
moderation operating points, explanations).

This repository holds the code and the aggregate results of every run. It does not hold the
manuscript, raw texts, row-level split manifests, per-example predictions, client partitions
or model weights. The split and every run can be regenerated from the original datasets.

The code is a research prototype. All parties run in one process without a network; each
check is made by the party that owns it, with keys and state held per party, and messages are
function calls. The results do not certify the security of a deployment.

## Protocol checks in the code

- A `SessionConfig` fixes members, dimension, sketch size, sketched coordinates, number of
  challenges, release threshold, attempt limit and update bound. Every authenticated message
  carries its hash.
- The auxiliary node enforces the order of the steps, releases mask sketches only of the agreed
  size, and releases one mask sum per attempt for a valid accepted set.
- Clients take part only in attempts that the auxiliary node opened, answer only for the set it
  authorised, derive the challenge themselves and answer once.
- Clients send their verification shares to both servers. Each server gives its own
  authenticated verdict; a client applies the aggregate only if it matches the commitment,
  lies in the integer range and both verdicts accept.

These checks refuse the malformed requests tested in `test_protocol.py` and `tamper.py`. They
harden the implementation; they are not a proof of confidentiality against a malicious server.

## Contents

| Path | Purpose |
|---|---|
| `secagg.py` | field arithmetic, encoding with integer bounds, ChaCha20 PRG with rejection sampling, CountSketch, clients, auxiliary node with release state, aggregation server |
| `fl.py` | head model, DP-SGD (CSPRNG sampling and noise), attacks, plaintext aggregators, `SecureRound` (one protocol attempt), training driver |
| `data_prep.py` | corpus preparation, label-conflict removal, split, manifest regeneration and hash check |
| `features.py` | frozen-encoder feature extraction (weights checked by SHA-256) |
| `exp.py` | experiment grids; every result stores input and code hashes; failed or stale runs give a non-zero exit status |
| `equivalence_check.py`, `legacy/`, `config/code_lineage.json` | the protocol code that produced the main grids, and a side-by-side test that the current code gives identical rounds and runs |
| `tune.py`, `moderation.py`, `sketch_fidelity.py`, `leak.py`, `xai.py`, `tamper.py`, `bench.py`, `near_dup_audit.py` | side studies |
| `test_secagg.py`, `test_protocol.py`, `test_dp.py`, `smoke_test.py` | unit, negative and boundary tests |
| `analyze_local.py` | full analysis of locally regenerated results |
| `analyze.py` | summary of the published aggregate results (standard library only) |
| `release_results.py`, `release_check.py` | aggregate-only export and content check |
| `config/defaults.json` | training configuration |
| `public_results/` | aggregate results of every run |
| `MANIFEST_SHA256.txt`, `PROVENANCE.json` | hash of the split manifest; hashes of the datasets and encoder files |
| `requirements.txt`, `requirements-lock.txt` | direct dependencies; complete lock of the stack used for the reported runs |

## Inspect the aggregate results

No datasets or training libraries are needed:

```bash
python release_check.py
python analyze.py --input public_results --output summary/aggregate_summary.csv
```

`analyze.py` recomputes accuracy, macro-F1, TMR and FAR of every run from its confusion matrix
and checks them against the stored values.

## Regenerate the experiments

Python 3.13, CPU only. Install `requirements-lock.txt` in a virtual environment. Obtain the
inputs under their original terms and place them as follows:

- `data/cyberbullying_tweets.csv`: Kaggle, Cyberbullying Classification;
- `data/dataset.json` and `data/post_id_divisions.json`: HateXplain (github.com/hate-alert/HateXplain, `Data/`);
- `model/`: `config.json`, `model.safetensors`, `tokenizer_config.json` and `vocab.txt` of `distilbert-base-uncased`.

`PROVENANCE.json` lists the SHA-256 hash of every input file.

```bash
python data_prep.py      # rebuilds the split; exits with an error if the manifest hash differs
python features.py
python test_secagg.py && python test_protocol.py && python test_dp.py
python smoke_test.py
./run_all.sh             # all grids and side studies (about 15 hours on 2 CPU cores)
python analyze_local.py
```

Masks, coins and DP noise come from the operating system's random source, so secure and DP runs
are not bit-for-bit repeatable. Plaintext runs without DP are.

## Provenance

Main-grid results store, at run time, hashes of the split manifest, the Kaggle features, the
protocol and training code and the run configuration. They were produced with the code in
`legacy/`, before the party-local verdicts and request checks were added. `equivalence_check.py`
runs both versions on the same inputs with the same sketch seed and records the code lineage in
`config/code_lineage.json`; `exp.py` accepts these older records only if that lineage holds. The
hash of the normaliser was added later and is checked retrospectively for those runs. The
integrity, cost and leakage results were produced with the current code and carry their own
provenance, written at run time. A cached leakage job is reused only if its stored provenance and
study settings (shadow clients, shadow federations, observed rounds, seed) match and all its
observation points are present. `tune.py`, `moderation.py` and `sketch_fidelity.py` now write a
`*_provenance.json` file next to their outputs. `xai.py` records provenance with each trained
head and retrains a head whose provenance differs; each explanation output carries a cache key
(`*.key`) built from its head, its attribution settings (integration steps, evaluation sample,
budget, bins) and the explanation code, and is recomputed when the key changes. The published
design-check, moderation, sketch-fidelity and explanation results were produced before these
records existed. Every runner checks the raw datasets and encoder files against
`PROVENANCE.json` at start-up and stops on a mismatch.

## Label-conflict policy

All eligible tweets (non-empty, at least three tokens) are normalised before any duplicate is
removed. Every group of identical normalised texts that carries more than one label is removed
completely. Each remaining group keeps its first tweet. `data_prep.py` writes a row-level audit
trail (`data/kaggle_group_audit_v3.csv`) that maps every eligible raw row to its group and decision.

## Privacy model of the DP runs

Zero-out adjacency with fixed public client capacities: neighbouring datasets differ in one
record slot of one client, which holds a real record in one and a null record (zero gradient)
in the other. Sampling and noise use the OS CSPRNG; floating-point Gaussian sampling is not
hardened against precision attacks. Client partitions are drawn from labels for simulation only.

## License and citation

MIT License (`LICENSE`). Citation metadata is in `CITATION.cff`.
