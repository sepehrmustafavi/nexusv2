"""Run the eight exploratory supervised experiments, resumably."""
import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .common import (AffectiveFeatures, digest, file_sha, latest_snapshot,
                     load_snapshot, normalize, prune_snapshots, save_snapshot,
                     seed_all, split_rows, write_json)
from .model import PilotClassifier


def experiment_grid(seed):
    return [dict(id=f'{m}-kg{int(k)}-lr{lr:g}-s{seed}', method=m,
                 knowledge=k, lr=lr, seed=seed)
            for i in range(2) for k in (False, True) for m in ('flex', 'lora')
            for lr in [({'flex': [2e-5, 1e-4], 'lora': [1e-4, 4e-4]})[m][i]]]


def source_digest():
    return digest({p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def prepare(args, root):
    path = root / 'protocol.json'
    settings = {k: getattr(args, k) for k in (
        'model', 'seed', 'epochs', 'max_length', 'micro_batch', 'accumulation',
        'eval_batch', 'warmup', 'weight_decay', 'head_lr', 'rank', 'alpha',
        'checkpoint_steps', 'precision', 'train_limit')}
    settings['source_digest'] = source_digest()
    settings['package_versions'] = {n: importlib.metadata.version(n)
        for n in ('transformers','peft','datasets','senticnet','numpy','scikit-learn','accelerate')}
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol['settings'] != settings:
            raise ValueError('Protocol/code changed. Use a new --output directory; never mix results.')
        if file_sha(root / 'data.json') != protocol['data_sha256']:
            raise ValueError('Prepared data changed/corrupted')
        return protocol, json.loads((root / 'data.json').read_text())
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    model_sha = api.model_info(args.model).sha
    dataset_sha = api.dataset_info('stanfordnlp/sst2').sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_sha)
    ds = load_dataset('stanfordnlp/sst2', revision=dataset_sha)
    train, tune = split_rows(list(ds['train']), list(ds['validation']), args.seed)
    if args.train_limit:
        # Debug only. A distinct output/protocol prevents mixing with full runs.
        from sklearn.model_selection import train_test_split
        train, _ = train_test_split(train, train_size=args.train_limit,
            random_state=args.seed, stratify=[r['label'] for r in train])
    features = AffectiveFeatures()
    data = {}
    for name, rows in [('train', train), ('tune', tune), ('report', list(ds['validation']))]:
        prepared = []
        for r in tqdm(rows, desc=f'Preparing {name}'):
            ids = tokenizer(r['sentence'], truncation=True, max_length=args.max_length)['input_ids']
            # Knowledge sees exactly the truncated text available to the model.
            visible = tokenizer.decode(ids, skip_special_tokens=True)
            score, matches = features.extract(visible)
            prepared.append(dict(id=int(r['idx']), input_ids=ids, label=int(r['label']),
                sentence=r['sentence'], symbolic_score=score, concepts=matches))
        data[name] = prepared
    write_json(root / 'data.json', data)
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'datasets', 'accelerate', 'senticnet', 'numpy', 'scikit-learn'):
        versions[name] = importlib.metadata.version(name)
    protocol = dict(settings=settings, model_revision=model_sha, dataset_revision=dataset_sha,
        data_sha256=file_sha(root / 'data.json'), versions=versions,
        knowledge_resource='SenticNet 6, 200000 concepts, packaged by senticnet 1.6 (not SenticNet 7)',
        lexicon_sha256=digest(features.lexicon), lexicon_size=len(features.lexicon),
        grid=experiment_grid(args.seed), split_counts={k: len(v) for k,v in data.items()},
        knowledge_coverage={k: float(np.mean([bool(r['concepts']) for r in v])) for k,v in data.items()},
        task='supervised SST-2; no unlabeled domain-adaptive pretraining',
        note='Official validation is a pilot reporting set, not an untouched future benchmark test.')
    write_json(path, protocol)
    return protocol, data


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([{'input_ids': r['input_ids'],
                                    'attention_mask': [1]*len(r['input_ids'])} for r in rows],
                                   padding=True, return_tensors='pt')
        batch['symbolic_score'] = torch.tensor([r['symbolic_score'] for r in rows], dtype=torch.float32)
        batch['labels'] = torch.tensor([r['label'] for r in rows])
        return batch


def loader(rows, tokenizer, size):
    # DataLoader iterator construction must not consume the model/dropout RNG.
    return DataLoader(rows, batch_size=size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(1234), collate_fn=Collate(tokenizer))


def amp_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == 'cuda' and precision == 'fp16')


def evaluate(model, rows, tokenizer, args, device):
    model.eval()
    probs, labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader(rows, tokenizer, args.eval_batch):
            batch = {k: v.to(device) for k,v in batch.items()}
            y = batch.pop('labels')
            with amp_context(device, args.precision):
                logits = model(**batch)
            logits = logits.float()
            losses.extend(torch.nn.functional.cross_entropy(logits, y, reduction='none').cpu().tolist())
            probs.extend(logits.softmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    predictions = np.argmax(probs, axis=1)
    metrics = dict(accuracy=float(accuracy_score(labels, predictions)),
                   macro_f1=float(f1_score(labels, predictions, average='macro', zero_division=0)),
                   positive_f1=float(f1_score(labels, predictions, zero_division=0)),
                   loss=float(np.mean(losses)),
                   confusion_matrix=confusion_matrix(labels, predictions, labels=[0,1]).tolist())
    return metrics, probs


def build_model(args, protocol, config, device):
    seed_all(args.seed)
    backbone = AutoModel.from_pretrained(args.model, revision=protocol['model_revision'],
        torch_dtype=torch.float32, attn_implementation='eager')
    backbone.config.use_cache = False
    model = PilotClassifier(backbone, config['method'], config['knowledge'],
                            seed=args.seed, rank=args.rank, alpha=args.alpha)
    # Non-reentrant checkpointing supports frozen embeddings in the LoRA arm.
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    return model


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def run_one(args, root, protocol, data, config, deadline):
    folder = root / config['id']
    if (folder / 'complete.json').exists():
        print('Already complete:', config['id'], flush=True)
        return True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('No GPU found. In Colab select Runtime > Change runtime type > T4 GPU.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=protocol['model_revision'])
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(args, protocol, config, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    metadata = dict(config=config, protocol_hash=digest(protocol), trainable=trainable,
        total=total, trainable_percent=100*trainable/total,
        device=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu')
    write_json(folder / 'run.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    head = list(model.head.parameters())
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': config['lr']},
                                  {'params': head, 'lr': args.head_lr}], weight_decay=args.weight_decay)
    effective = args.micro_batch * args.accumulation
    steps_epoch = math.ceil(len(data['train']) / effective)
    steps_total = args.epochs * steps_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_total*args.warmup), steps_total)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    epoch, offset, step, best, history, elapsed_total, peak = 0, 0, 0, -1., [], 0., 0
    _, state = latest_snapshot(folder)
    if state is not None:
        if state['protocol_hash'] != digest(protocol):
            raise ValueError('Checkpoint protocol mismatch')
        model.restore_trainable(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        epoch, offset, step, best = state['epoch'], state['offset'], state['step'], state['best']
        history, elapsed_total, peak = state['history'], state['elapsed_seconds'], state['peak_vram_bytes']
        restore_rng(state['rng'])
        del state
        print(f'Resuming epoch={epoch+1}, next example offset={offset}, optimizer step={step}', flush=True)
    else:
        seed_all(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def checkpoint():
        current_peak = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
        payload = dict(protocol_hash=digest(protocol), model=model.trainable_state(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            epoch=epoch, offset=offset, step=step, best=best, history=history,
            elapsed_seconds=elapsed_total+time.monotonic()-started,
            peak_vram_bytes=max(peak,current_peak),
            rng={'python':random.getstate(), 'numpy':np.random.get_state(),
                 'torch':torch.get_rng_state(),
                 'cuda':torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        save_snapshot(folder, f'step-{step:09d}', payload)
        prune_snapshots(folder)

    while epoch < args.epochs:
        order = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        model.train()
        progress = tqdm(total=steps_epoch, initial=offset//effective, desc=f"{config['id']} epoch {epoch+1}")
        while offset < len(order):
            window = [data['train'][i] for i in order[offset:offset+effective]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for batch in loader(window, tokenizer, args.micro_batch):
                batch = {k:v.to(device) for k,v in batch.items()}
                y = batch.pop('labels')
                with amp_context(device, args.precision):
                    logits = model(**batch)
                    loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction='sum') / len(window)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; no success result will be written.')
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            step += 1
            offset += len(window)
            progress.update(1)
            progress.set_postfix(loss=f'{loss_sum:.3f}', grad=f'{float(norm):.2f}')
            if step % args.checkpoint_steps == 0:
                checkpoint()
            if time.monotonic() >= deadline:
                checkpoint()
                progress.close()
                print('Session time budget reached. Rerun the same command to resume.', flush=True)
                return False
        progress.close()
        metrics, _ = evaluate(model, data['tune'], tokenizer, args, device)
        history.append(dict(epoch=epoch+1, step=step, tune=metrics))
        write_json(folder / 'history.json', history)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            save_snapshot(folder, f'best-epoch-{epoch+1}', dict(model=model.trainable_state(), tune=metrics, epoch=epoch+1))
        epoch += 1
        offset = 0
        checkpoint()
    valid_best = []
    for path in folder.glob('best-epoch-*.pt'):
        try:
            saved = load_snapshot(path)
            valid_best.append((saved['tune']['macro_f1'], path))
            del saved
        except (OSError, ValueError, EOFError, RuntimeError, json.JSONDecodeError):
            continue
    if not valid_best:
        raise RuntimeError('No valid best checkpoint remains')
    _, best_path = max(valid_best, key=lambda pair: pair[0])
    best_state = load_snapshot(best_path)
    model.restore_trainable(best_state['model'])
    metrics = best_state['tune']
    elapsed = elapsed_total + time.monotonic()-started
    peak = max(peak, torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
    # Completion is committed before old resumable snapshots are removed.
    write_json(folder / 'complete.json', dict(**metadata, tune=metrics,
        selected_epoch=best_state['epoch'], best_checkpoint=best_path.name, elapsed_seconds=elapsed,
        peak_vram_bytes=peak, selection_set='internal tune only'))
    prune_snapshots(folder, keep=0)
    for path in folder.glob('best-epoch-*.pt'):
        if path != best_path:
            path.unlink()
            path.with_suffix('.json').unlink(missing_ok=True)
    del model, optimizer, scheduler, best_state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return True


def finalize(args, root, protocol, data):
    """Report only after all eight candidates finish; select LR using tune only."""
    candidates = []
    for config in protocol['grid']:
        p = root / config['id'] / 'complete.json'
        if not p.exists():
            raise RuntimeError(f"Suite incomplete: {config['id']}. Run again without --finalize.")
        candidates.append(json.loads(p.read_text()))
    write_json(root / 'all_candidates.json', candidates)
    selected = []
    for method in ('flex', 'lora'):
        for knowledge in (False,True):
            pool = [r for r in candidates if r['config']['method']==method and r['config']['knowledge']==knowledge]
            selected.append(max(pool, key=lambda r:r['tune']['macro_f1']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('Select a GPU runtime for final evaluation.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=protocol['model_revision'])
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    summaries = []
    for record in selected:
        config = record['config']
        folder = root / config['id']
        model = build_model(args, protocol, config, device)
        model.restore_trainable(load_snapshot(folder / record['best_checkpoint'])['model'])
        metrics, probs = evaluate(model, data['report'], tokenizer, args, device)
        with (folder / 'predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id','label','prediction','p_negative','p_positive','symbolic_score','concepts','sentence'])
            writer.writeheader()
            for r,p in zip(data['report'],probs):
                writer.writerow(dict(id=r['id'],label=r['label'],prediction=int(np.argmax(p)),
                    p_negative=p[0],p_positive=p[1],symbolic_score=r['symbolic_score'],
                    concepts=json.dumps(r['concepts']),sentence=r['sentence']))
        summary = dict(method=config['method'], knowledge=config['knowledge'], selected_lr=config['lr'],
            selected_epoch=record['selected_epoch'], tune_macro_f1=record['tune']['macro_f1'],
            report_accuracy=metrics['accuracy'], report_macro_f1=metrics['macro_f1'],
            report_positive_f1=metrics['positive_f1'], trainable=record['trainable'],
            trainable_percent=record['trainable_percent'], peak_vram_gib=record['peak_vram_bytes']/2**30,
            elapsed_hours=record['elapsed_seconds']/3600)
        summaries.append(summary)
        del model
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    write_json(root / 'summary.json', summaries)
    with (root / 'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    effects = {}
    for method in ('flex','lora'):
        off, on = [next(r for r in summaries if r['method']==method and r['knowledge']==k) for k in (False,True)]
        effects[method+'_knowledge_delta_macro_f1_pp'] = 100*(on['report_macro_f1']-off['report_macro_f1'])
    effects['caution'] = 'Single seed, separately tuned candidates. Descriptive pilot deltas; no significance or causal interaction claim.'
    write_json(root / 'effects.json', effects)
    print(json.dumps({'summary': summaries, 'effects':effects},indent=2))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='Qwen/Qwen2-0.5B')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--micro-batch',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--warmup',type=float,default=.1)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=int,default=16)
    p.add_argument('--checkpoint-steps',type=int,default=100)
    p.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--max-hours',type=float,default=2.)
    p.add_argument('--train-limit',type=int,default=0,help='Debug subset only; use a separate output directory')
    p.add_argument('--only',default='',help='Optional exact run id from --list')
    p.add_argument('--list',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--finalize',action='store_true')
    p.add_argument('--allow-cpu',action='store_true',help='Debug only; real Qwen training on CPU is impractical')
    args=p.parse_args()
    for name in ('epochs','max_length','micro_batch','accumulation','eval_batch','checkpoint_steps'):
        if getattr(args,name)<1: p.error(name+' must be positive')
    if args.max_hours <= 0: p.error('--max-hours must be positive')
    return args


def main():
    args=parse_args()
    if args.list:
        print(json.dumps(experiment_grid(args.seed),indent=2)); return
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    # Runtime version drift must be visible even when resuming a frozen protocol.
    write_json(root / 'latest_environment.json', dict(python=os.sys.version,
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','datasets','senticnet')},
        cuda=torch.version.cuda))
    protocol,data=prepare(args,root)
    if args.prepare_only: return
    if args.finalize:
        finalize(args,root,protocol,data); return
    deadline=time.monotonic()+args.max_hours*3600
    grid=[c for c in protocol['grid'] if not args.only or c['id']==args.only]
    if not grid: raise ValueError('Unknown --only id; inspect --list')
    for config in grid:
        if time.monotonic()>=deadline or not run_one(args,root,protocol,data,config,deadline): break
    done=sum((root/c['id']/'complete.json').exists() for c in protocol['grid'])
    print(f'Completed {done}/8 candidates. No official-validation selection was performed.')
    if done==8: print('Next: rerun the same command with --finalize to generate summary.csv and predictions.')


if __name__=='__main__':
    main()
