# Paper datasets

Run the commands from the repository root. To prepare all seven tasks or check a prepared task offline:

```bash
python data/prepare.py --task all
python data/prepare.py --task trec --verify-only
```

The first command downloads the source files at the revisions recorded in [sources.json](sources.json). It checks each file's SHA-256, builds a Hugging Face `DatasetDict` under `data/prepared/<task>/`, and verifies the count, content and order of the resulting records against the experiment snapshot. An existing task directory is verified on rerun. Each linked source provides its own distribution terms and attribution.

| Task | Source | Train pool | Full evaluation split | Evaluation examples |
|---|---|---:|---|---:|
| SST-2 | [GLUE](https://huggingface.co/datasets/nyu-mll/glue) | 67,349 | Validation | 872 |
| TREC | [SetFit/TREC-QC](https://huggingface.co/datasets/SetFit/TREC-QC) | 5,441 | Test, six coarse classes | 500 |
| CR | [SetFit/CR](https://huggingface.co/datasets/SetFit/CR) | 3,394 | Test | 376 |
| CoLA | [GLUE](https://huggingface.co/datasets/nyu-mll/glue) | 8,535 | Validation | 1,043 |
| RTE | [SuperGLUE](https://huggingface.co/datasets/aps/super_glue) | 2,490 | Validation | 277 |
| MNLI | [GLUE](https://huggingface.co/datasets/nyu-mll/glue) | 392,702 | Matched and mismatched validation | 19,647 |
| SUBJ | [SetFit/subj](https://huggingface.co/datasets/SetFit/subj) | 8,000 | Test | 2,000 |

**Training and evaluation.** Each task exposes `train` and `validation` on disk. The runner takes 1,000 training examples followed by 100 development examples from a fixed NumPy permutation of the training pool (data seed 0). It uses the 100 development examples for checkpoint selection. The `validation` directory stores the full evaluation split shown above. For MNLI, the 9,815 matched examples precede the 9,832 mismatched examples.

**Preprocessing.** SST-2 and RTE preserve the source records. The remaining five tasks expose `idx`, `text`, `text_pair`, `label` and `eval_split`. TREC's integer labels, in order, correspond to ABBR, ENTY, DESC, HUM, LOC and NUM. For CoLA and TREC we remove 16 and 11 training rows respectively whose normalized text also appears in the full evaluation split. Normalization lowercases and collapses whitespace for matching. The runner skips repeated normalized training inputs when sampling CoLA, CR, SUBJ, TREC and MNLI.

**Offline use.** Place the files listed in the manifest under `<mirror>/<repository with "/" replaced by "__">/<file>`. For example, the TREC training file is `<mirror>/SetFit__TREC-QC/train.jsonl`. Then run:

```bash
python data/prepare.py --task trec --raw-root /path/to/mirror
```
