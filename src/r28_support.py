"""Schedule arithmetic and observation helpers; no optimizer changes."""
import math


def effective_lr(base_lr, step, drop_after=0, factor=.25):
    if step < 1 or drop_after < 0 or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('Invalid step/drop policy')
    return base_lr * (factor if drop_after and step > drop_after else 1.)


def expected_lr(initial_lr, step, drop_after=0, factor=.25):
    return effective_lr(initial_lr * max(0., (10001-step)/10000), step, drop_after, factor)


def observe_accuracy(model, scores, labels):
    """Reuse the existing classification scores only during development evaluation."""
    counts = getattr(model, '_r28_eval_counts', None)
    if counts is not None:
        counts[0].add_((scores.argmax(dim=-1) == labels).sum())
        counts[1].add_(labels.numel())
