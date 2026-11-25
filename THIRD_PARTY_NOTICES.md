# Third-party notices

The training implementation incorporates the following projects. Copies of their license texts accompany the source.

| Project | Code used in this repository | License |
| --- | --- | --- |
| [FZOO](https://github.com/DKmiyan/FZOO) | LLM training framework, FZOO, tasks and templates | [MIT](LICENSES/FZOO-MIT.txt) |
| [MeZO](https://github.com/princeton-nlp/MeZO) | Underlying forward-only LLM training framework | [MIT](LICENSES/MeZO-MIT.txt) |
| [R-AdaZO](https://github.com/shuyao95/R-AdaZO) | Reference optimizer and adaptive moment update | [MIT](LICENSES/R-AdaZO-MIT.txt) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Trainer and model code in `trainer.py`, `mezotrainer.py` and `ParallelOPTForCausalLM.py` | [Apache License 2.0](LICENSES/Transformers-Apache-2.0.txt) |

The trainers include research modifications for forward-only optimization, diagnostics and evaluation. The upstream copyrights remain in the source. The notices in `src/LICENSE` and `src/R_ADAZO_LICENSE` accompany the inherited code.

The modules in `src/` are copied from the training runtime used for the final MNLI experiment. Their original filenames are retained so that the code can be compared with the recorded runs. The file `scripts/runtime_sha256.json` contains their SHA-256 digests. The portable scripts select among the four moment representations, supply local data paths and validate the resulting runs. Dataset sources are listed in [the data guide](data/README.md). Model and dataset terms apply separately.
