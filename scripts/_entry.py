"""Route seven prepared datasets to the research trainer and its evaluation code."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--r22-artifacts", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.data_root.resolve()
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))
    import tasks
    import tasks41
    import metrics41
    import run
    import r39_train

    load_from_disk = tasks.load_from_disk

    def load_prepared(path, *positional, **keywords):
        return load_from_disk(str(root / Path(path).name), *positional, **keywords)

    tasks.load_from_disk = load_prepared
    tasks41.install(root)
    metrics41.install(run, args.r22_artifacts)
    sys.argv = [str(src / "r39_train.py"), "--r22-artifacts", str(args.r22_artifacts), *remaining]
    r39_train.main()


if __name__ == "__main__":
    main()
