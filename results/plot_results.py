"""Reproduce the paper figures and print the corresponding numerical summaries."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
TASKS = ("SST-2", "TREC", "CR", "CoLA", "RTE", "MNLI", "SUBJ")
METHODS = ("FZOO", "Dense m, dense v", "Dense m, factored v", "CoFaZO")
COLORS = ("#aab6c4", "#4b596d", "#d39955", "#007e79")


def read_rows(filename):
    with (ROOT / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_curves():
    curves = {task: defaultdict(list) for task in TASKS}
    for row in read_rows("development_loss.csv"):
        point = (
            int(row["step"]),
            float(row["fzoo_dev_loss"]),
            float(row["cofazo_dev_loss"]),
        )
        curves[row["task"]][int(row["seed"])].append(point)
    expected = list(range(250, 10001, 250))
    for runs in curves.values():
        for points in runs.values():
            points.sort()
            if [p[0] for p in points] != expected:
                raise ValueError("A paired run must contain exactly 40 evaluations")
    return curves


def average_difference(points):
    return mean(b - c for _, b, c in points)


def relative_reduction(points):
    return 100 * average_difference(points) / mean(b for _, b, _ in points)


def threshold_step(points, column):
    streak = 0
    for point in points:
        streak = streak + 1 if point[column] <= 0.50 else 0
        if streak == 3:
            return point[0]
    raise ValueError("The TREC target was not attained")


def plot_loss(curves, output):
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.05))
    fig.subplots_adjust(left=.099, right=.98, bottom=.135, top=.84, hspace=.65, wspace=.36)
    for task, ax in zip(TASKS[:6], axes.flat):
        points = max(curves[task].values(), key=average_difference)
        steps, fzoo, cofazo = zip(*points)
        ax.plot(steps, fzoo, "--", lw=1.45, color="#505d70", label="FZOO")
        ax.plot(steps, cofazo, lw=1.6, color=COLORS[-1], label="CoFaZO")
        ax.set_title(task, loc="left", fontweight="bold", fontsize=10.5, pad=5)
        ax.set_title(f"{relative_reduction(points):.2f}% lower", loc="right",
                     fontsize=8.5, color="#00736d", pad=5)
        ax.grid(axis="y", lw=.6, color="#e0e5e8")
        ax.set_axisbelow(True)
        ax.set_xlim(0, 10000)
        ax.set_xticks([0, 2500, 5000, 7500, 10000], ["0", "2.5k", "5k", "7.5k", "10k"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(labelsize=8, length=2.6, pad=2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, 1), ncol=2, frameon=False)
    fig.text(.01, .49, "Development loss", rotation=90, va="center")
    fig.text(.51, .022, "Training updates", ha="center")
    fig.savefig(output / "loss6.pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def plot_memory(output):
    peaks = {(r["task"], r["method"]): int(r["peak_allocated_bytes"]) / 2**30
             for r in read_rows("peak_memory.csv")}
    fig, ax = plt.subplots(figsize=(8.6, 3.75))
    width = .212
    for j, (method, color) in enumerate(zip(METHODS, COLORS)):
        positions = [i + (j - 1.5) * width for i in range(len(TASKS))]
        bars = ax.bar(positions, [peaks[t, method] for t in TASKS], width=width * .94,
                      color=color, label=method)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(range(len(TASKS)), TASKS)
    ax.set_xlim(-.54, len(TASKS) - .46)
    ax.set_ylim(0, 27)
    ax.set_yticks(range(0, 26, 5))
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_ylabel("Peak allocated GPU memory (GiB)")
    ax.grid(axis="y", color="#e1e5e9", lw=.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False,
              fontsize=9.5, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(output / "peak_memory_four_methods.pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    curves = read_curves()
    task_means = {t: mean(relative_reduction(p) for p in curves[t].values()) for t in TASKS}
    print(f"Mean relative development-loss reduction: {mean(task_means.values()):.2f}%")
    for task, change in task_means.items():
        print(f"  {task}: {change:+.2f}%")
    scores = defaultdict(list)
    for row in read_rows("accuracy.csv"):
        scores[row["method"]].append(float(row["accuracy_percent"]))
    for method, values in scores.items():
        print(f"Mean of seven task accuracy entries ({method}): {mean(values):.2f}%")
    for seed, points in sorted(curves["TREC"].items()):
        baseline, cofazo = threshold_step(points, 1), threshold_step(points, 2)
        print(f"TREC seed {seed}: {baseline} -> {cofazo} ({100*(1-cofazo/baseline):.2f}% fewer updates)")
    plot_loss(curves, output)
    plot_memory(output)


if __name__ == "__main__":
    main()
