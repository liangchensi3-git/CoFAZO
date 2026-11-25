"""New dataset/answer adapters. No changes to the inherited task classes."""
from pathlib import Path
import numpy as np
from datasets import load_from_disk
import tasks
from utils import temp_seed

NAMES = {'cola': 'CoLA', 'cr': 'CR', 'subj': 'SUBJ', 'trec': 'TREC',
         'mrpc': 'MRPC', 'snli': 'SNLI', 'mnli': 'MNLI'}
ANSWERS = {'cola': ('incorrect', 'correct'), 'cr': ('terrible', 'great'),
           'subj': ('objective', 'subjective'),
           'trec': ('an abbreviation', 'an entity', 'a description', 'a person', 'a location', 'a number'),
           'mrpc': ('No', 'Yes'), 'snli': ('Yes', 'Maybe', 'No'), 'mnli': ('Yes', 'Maybe', 'No')}


def text_key(row):
    return tuple(' '.join(row[k].lower().split()) for k in ('text', 'text_pair'))


class SampleSequence:
    """Read only requested Arrow rows instead of materializing 550k Samples."""
    def __init__(self, data, build):
        self.data, self.build = data, build

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.build(self.data[index])


class ClassificationTemplate:
    def __init__(self, key):
        self.key = key

    def encode(self, sample):
        x = sample.data
        if self.key == 'cr':
            return x['text'].strip() + ' It was'
        if self.key == 'cola':
            return x['text'].strip() + '\nThis sentence is grammatically'
        if self.key == 'subj':
            return x['text'].strip() + '\nThis text is'
        if self.key == 'trec':
            return x['text'].strip() + '\nThe question asks about'
        if self.key == 'mrpc':
            return 'Sentence 1: ' + x['text'] + '\nSentence 2: ' + x['text_pair'] + '\nDo these sentences have the same meaning?'
        return 'Suppose ' + x['text'] + ' Can we infer that "' + x['text_pair'] + '"? Yes, No, or Maybe?'

    def verbalize(self, sample, candidate):
        return self.encode(sample) + ' ' + ANSWERS[self.key][candidate]

    def encode_sfc(self, sample):
        return ''

    def verbalize_sfc(self, sample, candidate):
        return ANSWERS[self.key][candidate]


def install(data_root=None):
    root = Path(data_root) if data_root else Path(__file__).resolve().parent / 'datasets'
    for key, name in NAMES.items():
        def create(key=key):
            class ClassificationDataset(tasks.Dataset):
                r41_key = key

                def __init__(self, subtask=None, **kwargs):
                    super().__init__(subtask, **kwargs)
                    data = load_from_disk(str(root / key))
                    self.samples = {target: SampleSequence(data[source], self.build_sample)
                                    for target, source in [('train', 'train'), ('valid', 'validation')]}

                def build_sample(self, row):
                    label = int(row['label'])
                    if label not in range(len(ANSWERS[key])):
                        raise ValueError('Invalid label: ' + key)
                    return tasks.Sample(id=row['idx'], data=row, correct_candidate=label,
                                        candidates=list(range(len(ANSWERS[key]))))

                def get_template(self, template_version=0):
                    if template_version != 0:
                        raise ValueError('Only the frozen task template is supported')
                    return ClassificationTemplate(key)

                def sample_subset(self, data_split='train', seed=0, num=100, exclude=None):
                    if data_split != 'train':
                        return super().sample_subset(data_split, seed, num, exclude)
                    if exclude is not None:
                        raise ValueError('Per-example training sets are outside this protocol')
                    # Retain seed0 ordering, but do not put identical input text
                    # into both training and the train-held-out development set.
                    samples = self.samples['train']
                    with temp_seed(seed):
                        order = np.random.permutation(len(samples)).tolist()
                    result, seen = [], set()
                    for i in order:
                        sample = samples[i]
                        fingerprint = text_key(sample.data)
                        if fingerprint in seen:
                            continue
                        result.append(sample); seen.add(fingerprint)
                        if len(result) == num:
                            return result
                    raise ValueError('Insufficient distinct training inputs')
            return ClassificationDataset
        setattr(tasks, name + 'Dataset', create())
