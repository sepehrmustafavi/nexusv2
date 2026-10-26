import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def normalize(text):
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text.lower()))


def split_rows(rows, external, seed=42, fraction=0.05):
    """Group exact normalized duplicates; exclude official-dev exact overlaps.

    SST-2 also contains related phrases; exact grouping does NOT remove every
    source-review/phrase relation. This remains an exploratory split.
    """
    from sklearn.model_selection import train_test_split
    forbidden = {normalize(row['sentence']) for row in external}
    groups = {}
    for row in rows:
        key = normalize(row['sentence'])
        if not key or key in forbidden:
            continue
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    # Majority label is used solely to stratify groups, not to change labels.
    labels = [int(np.mean([r['label'] for r in groups[k]]) >= .5) for k in keys]
    train_keys, tune_keys = train_test_split(
        keys, test_size=fraction, random_state=seed, stratify=labels)
    flatten = lambda ks: [r for k in ks for r in groups[k]]
    return flatten(train_keys), flatten(tune_keys)


class AffectiveFeatures:
    """Greedy longest-match SenticNet concepts within each word segment.

    This is a documented pilot operationalization, not a claim of exact
    replication of the old manuscript's unspecified keyword extractor.
    No negation correction or contextual word-sense disambiguation is claimed.
    """
    def __init__(self, lexicon=None, max_ngram=5):
        if lexicon is None:
            from senticnet.senticnet import SenticNet
            sn = SenticNet()
            lexicon = {k: float(sn.polarity_value(k)) for k in sn.data}
        self.lexicon = lexicon
        self.max_ngram = max_ngram

    def extract(self, text):
        matches = []
        # Do not match multiword concepts across punctuation boundaries.
        for segment in re.split(r'[.!?;,:\n]+', text):
            words = normalize(segment).split()
            i = 0
            while i < len(words):
                for n in range(min(self.max_ngram, len(words)-i), 0, -1):
                    concept = '_'.join(words[i:i+n])
                    if concept in self.lexicon:
                        matches.append({'concept': concept, 'polarity': self.lexicon[concept]})
                        i += n
                        break
                else:
                    i += 1
        # Match the set-based concept aggregation in the manuscript.
        unique = {m['concept']: m for m in matches}
        matches = list(unique.values())
        return (float(np.mean([m['polarity'] for m in matches])) if matches else 0.0), matches


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Best-effort reproducibility, not a promise of cross-GPU bitwise identity.
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_snapshot(folder, name, payload):
    """A snapshot is usable only after its completion/hash marker is present."""
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + '.pt')
    temp = folder / (name + '.pt.tmp')
    torch.save(payload, temp)
    checksum = file_sha(temp)
    temp.replace(target)
    write_json(target.with_suffix('.json'), {'sha256': checksum})
    return target


def load_snapshot(path, device='cpu'):
    import torch
    path = Path(path)
    marker = json.loads(path.with_suffix('.json').read_text())
    if file_sha(path) != marker['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    # Only load locally generated checkpoints in your own output directory.
    return torch.load(path, map_location=device, weights_only=False)


def latest_snapshot(folder):
    for p in sorted(Path(folder).glob('step-*.pt'), reverse=True):
        try:
            return p, load_snapshot(p)
        except (OSError, ValueError, EOFError, RuntimeError, KeyError, json.JSONDecodeError):
            continue
    if list(Path(folder).glob('step-*')):
        raise RuntimeError('No complete checkpoint remains. Preserve the directory and inspect it; do not silently restart.')
    return None, None


def prune_snapshots(folder, keep=2):
    complete = [p for p in sorted(Path(folder).glob('step-*.pt'), reverse=True)
                if p.with_suffix('.json').exists()]
    for p in complete[keep:]:
        p.unlink()
        p.with_suffix('.json').unlink()
