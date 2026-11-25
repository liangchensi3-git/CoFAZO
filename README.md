# CoFaZO

**Compact Moment States for Full-Parameter Zeroth-Order Fine-Tuning**

Liangchen Si · [Manuscript (PDF)](paper/CoFaZO.pdf) · [Experimental results](results/) · [Citation](CITATION.cff)

CoFaZO adds adaptive moment updates to the forward-only FZOO estimator while keeping the optimizer state small enough for full-parameter fine-tuning. It packs each matrix momentum entry into four bits and stores the second moment as row and column averages of squared momentum. Both the weight update and the second-moment update use the current momentum in FP32 before its history is compressed.

## Results

The manuscript studies seven classification tasks on Qwen3-1.7B-Base. With pure FZOO as the reference, measured peak GPU memory ranges from **2.13–3.72×** for dense FP32 first and second moments to **1.07–1.17×** for CoFaZO. These figures describe peak allocated memory for the measured workloads, including the model and temporary allocations.

Across 19 paired runs, CoFaZO reduced mean development loss by **4.90%**, averaging the relative reductions within each task and then weighting the seven tasks equally. On TREC, the first three consecutive development evaluations with loss at or below 0.50 were completed **18.8–29.4%** earlier in update count. The [results directory](results/) contains the measurements, selection rules and scripts for the figures.

To regenerate the loss and memory figures directly from the included CSVs, run the following commands on a CPU.

```shell
python -m pip install matplotlib
python results/plot_results.py
```

## Reproduce a run

Use Python 3.10 and install a PyTorch build compatible with your GPU first. The reference training environment used PyTorch 2.4.1 with CUDA 12.1. The commands below install the remaining dependencies, prepare the data and run SST-2 from the repository root.

```shell
python -m pip install -r requirements.txt
python data/prepare.py --task sst2 --output data/prepared
python scripts/train.py --model /path/to/Qwen3-1.7B-Base --data-root data/prepared --dataset sst2 --method cofazo --learning-rate 1e-5 --seed 42 --output runs/sst2-cofazo
```

Model weights are available from [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base). Use `python data/prepare.py --task all` to prepare all seven tasks. The [data guide](data/README.md) documents the sources, split construction and verification. The other task identifiers are `trec`, `cr`, `cola`, `rte`, `mnli` and `subj`. For the accuracy table, use the settings in [accuracy.csv](results/accuracy.csv). For the convergence curves, use the paired learning rates and seeds in [development_loss.csv](results/development_loss.csv).

| Training option | Matrix momentum carried across steps | Matrix second moment carried across steps |
| --- | --- | --- |
| `fzoo` | None | None |
| `dense` | Dense FP32 | Dense FP32 |
| `factored` | Dense FP32 | FP32 row and column factors |
| `cofazo` | Four-bit codes and FP32 row and column scales | FP32 row and column factors |

The paper uses 10,000 updates, batch size 16 and a development evaluation every 250 updates. Each update uses one unperturbed forward query and eight perturbed queries on the same minibatch. To check the implementation locally, run `python scripts/run_tests.py`.

## How the state is stored

For a matrix with $a$ rows and $b$ columns, a first pass over parameter chunks decodes the previous momentum and combines it with the current direction. Row and column means of $|M_t|$ set the new encoding scales. Row and column means of $M_t^2$ update the second-moment factors. A second pass regenerates the same direction chunks from their seeds, applies the weight update and packs the new history. The old encoding scales stay in place until every old code in that matrix has been read. The second pass uses local arithmetic and makes no additional model queries.

| Matrix state representation | Persistent optimizer bytes |
| --- | --- |
| Dense FP32 moments | $8ab$ |
| Dense FP32 momentum and factored second moment | $4ab + 4(a+b)$ |
| CoFaZO | $a\lceil b/2\rceil + 8(a+b)$ |

The last row includes one sign bit and three magnitude bits per element, row-wise packing into bytes, and two pairs of FP32 vectors. One-dimensional parameters retain dense FP32 moments. These formulas concern optimizer state alone. The peak-memory results above cover the entire training process.

The implementation entry points are the [all-matrix momentum update](src/r39_all_m.py) and the [four-bit encoder and decoder](src/rc4_m.py). Runtime file checksums and upstream attributions are documented in the [third-party notices](THIRD_PARTY_NOTICES.md).
