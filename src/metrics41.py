"""Extra final-evaluation metrics from the existing predictions; no extra forwards."""
import hashlib
import json
import math
from pathlib import Path
import time


def scores(gold, pred, classes):
    if not gold or len(gold) != len(pred):
        raise ValueError('Missing or mismatched predictions')
    matrix = [[0] * classes for _ in range(classes)]
    for a, b in zip(gold, pred):
        if a not in range(classes) or b not in range(classes):
            raise ValueError('Invalid class index')
        matrix[a][b] += 1
    n = len(gold)
    result = {'accuracy': sum(matrix[i][i] for i in range(classes)) / n}
    f1s = []
    for i in range(classes):
        tp = matrix[i][i]
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1s.append(2 * tp / denominator if denominator else 0.)
    result['macro_f1'] = sum(f1s) / classes
    if classes == 2:
        tn, fp = matrix[0]; fn, tp = matrix[1]
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        result['f1'] = f1s[1]
        result['mcc'] = (tp * tn - fp * fn) / denominator if denominator else 0.
    return result


def install(run, output):
    original = run.Framework.evaluate
    destination = Path(output) / 'evaluations41.jsonl'

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        if not hasattr(self.task, 'r41_key'):
            return original(self, train_samples, eval_samples, one_train_set_per_eval_sample)
        from tasks41 import ANSWERS
        predictions = []
        bound = self.one_step_pred
        had_instance_method = 'one_step_pred' in self.__dict__
        def capture(*args, **kwargs):
            value = bound(*args, **kwargs)
            predictions.append(value)
            return value
        started = time.perf_counter()
        self.one_step_pred = capture
        try:
            legacy = original(self, train_samples, eval_samples, one_train_set_per_eval_sample)
        finally:
            if had_instance_method:
                self.one_step_pred = bound
            else:
                del self.one_step_pred
        gold = [int(p.correct_candidate) for p in predictions]
        pred = [int(p.predicted_candidate) for p in predictions]
        classes = len(ANSWERS[self.task.r41_key])
        result = scores(gold, pred, classes)
        if result['accuracy'] != legacy['accuracy']:
            raise ValueError('Prediction capture changed the original accuracy')
        splits = [s.data['eval_split'] for s in eval_samples]
        if self.task.r41_key == 'mnli':
            for split in ('matched', 'mismatched'):
                ids = [i for i, value in enumerate(splits) if value == split]
                if ids:
                    result[split + '_accuracy'] = sum(gold[i] == pred[i] for i in ids) / len(ids)
        record = dict(dataset=self.task.r41_key, count=len(gold), gold=gold, predicted=pred,
                      splits=splits, metrics=result, seconds=time.perf_counter() - started,
                      sample_hash=hashlib.sha256(json.dumps([s.data for s in eval_samples],
                          sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('a', encoding='utf8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        return result
    run.Framework.evaluate = evaluate
