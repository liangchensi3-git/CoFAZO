# Paper results

The paper evaluates full-parameter fine-tuning of Qwen3-1.7B-Base. The three CSVs contain the measurements behind the accuracy table, the loss curves and the GPU-memory comparison. Run `python results/plot_results.py` from the repository root with Matplotlib installed to regenerate both PDFs and print the numerical summaries.

## Classification accuracy

[`accuracy.csv`](accuracy.csv) contains one entry per method and task. For each run, the checkpoint with the lowest development loss was evaluated on the full evaluation split. Each table entry reports the highest such evaluation accuracy among the available runs for that method and task. The CSV records the selected run's seed, learning rate, training step and accuracy on a 0–100 scale. Methods can select different runs and have different numbers of candidates. CoLA is scored by accuracy. MNLI accuracy pools all 19,647 validation examples across the matched (9,815) and mismatched (9,832) splits.

| Method | SST-2 | TREC | CR | CoLA | RTE | MNLI | SUBJ | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FZOO | 93.46 | 91.40 | 91.22 | 78.72 | 85.56 | 83.35 | 93.00 | 88.10 |
| Dense m, dense v | 92.89 | 93.20 | 91.49 | 76.99 | 85.20 | 82.73 | 91.90 | 87.77 |
| CoFaZO | 93.23 | 93.00 | 92.29 | 77.37 | 85.56 | 83.59 | 93.70 | 88.39 |

## Development loss and convergence

[`development_loss.csv`](development_loss.csv) contains both methods' losses for 19 paired runs at fixed per-task learning rates. Each pair uses the same seed and contributes 40 evaluations at intervals of 250 steps through step 10,000. RTE and MNLI each have two pairs. The other five tasks each have three. The learning rates for both methods appear in each row.

For a pair, let `B` and `C` be its mean development losses over the 40 evaluations for FZOO and CoFaZO, respectively. We calculate `100 × (B − C) / B`, average these percentages across seeds within each task, then average the seven task results. This gives a **4.90%** mean reduction. The task-level reductions are SST-2 +4.72%, TREC +11.46%, CR +3.14%, CoLA +0.77%, RTE −0.20%, MNLI −1.11% and SUBJ +15.50%.

[The six-panel loss figure](figures/loss6.pdf) shows one pair per displayed task, selected by the largest signed difference `B − C`. Its annotations apply to the individual pairs shown. SUBJ contributes to the seven-task summary and its observations are available in the CSV.

On TREC, the convergence target is three consecutive evaluations with development loss at or below 0.50. Counting updates through the third evaluation, FZOO and CoFaZO respectively took 3,500 and 2,500 steps for seed 42, 4,000 and 3,250 for seed 43, and 4,250 and 3,000 for seed 44.

## GPU memory

[`peak_memory.csv`](peak_memory.csv) records the reported peak allocated GPU bytes, persistent optimizer-state bytes, hardware and seed for each task and method. [The four-method memory figure](figures/peak_memory_four_methods.pdf) converts the measured allocation peaks to GiB by dividing bytes by 2³⁰. FZOO is the forward-only baseline. The two intermediate comparisons store dense FP32 momentum and use either a dense FP32 second moment or FP32 row and column factors. CoFaZO stores four-bit matrix momentum history and the factored second moment.

Across the seven tasks, the peak relative to the task's FZOO allocation ranges from 2.13× to 3.72× with two dense moments, from 1.57× to 2.36× with dense momentum and factored second moments, and from 1.07× to 1.17× with CoFaZO. For TREC and CoLA, the two-dense-moment run used an L40S and the other three runs used an A100. The CSV identifies the measurements for each comparison.
