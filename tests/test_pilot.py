"""Offline checks using a randomly initialized, tiny Qwen2 (NOT benchmark results)."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2Model

from nexus_pilot.common import (AffectiveFeatures, latest_snapshot, load_snapshot,
                                normalize, save_snapshot, seed_all, split_rows)
from nexus_pilot.model import PilotClassifier, last_valid
from nexus_pilot.run import experiment_grid, finalize, run_one


class TinyTokenizer:
    pad_token_id = 0
    padding_side = 'right'

    def pad(self, rows, **kwargs):
        size=max(len(r['input_ids']) for r in rows)
        return {key:torch.tensor([r[key]+[0]*(size-len(r[key])) for r in rows])
                for key in ('input_ids','attention_mask')}


def tiny_model(args, protocol, config, device):
    seed_all(args.seed)
    backbone=Qwen2Model(Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=128,attention_dropout=0.,attn_implementation='eager'))
    return PilotClassifier(backbone,config['method'],config['knowledge'],seed=args.seed,
                           rank=args.rank,alpha=args.alpha).to(device)


def fixtures():
    args=SimpleNamespace(model='tiny-offline-test',seed=42,rank=2,alpha=4,
        precision='fp32',allow_cpu=True,epochs=2,micro_batch=2,accumulation=2,
        eval_batch=4,head_lr=.001,weight_decay=.01,warmup=.1,checkpoint_steps=1)
    rows=[dict(id=i,input_ids=[1,2+(i%20),3,4][:3+i%2],label=i%2,
               symbolic_score=(-.7 if i%2==0 else .8),concepts=[],sentence=str(i)) for i in range(12)]
    return args,dict(train=rows,tune=rows[:4],report=rows[4:8])


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_longest_match_and_missing_knowledge(self):
        f=AffectiveFeatures({'very_good':.9,'good':.7,'bad':-.8})
        score,m=f.extract('very good, bad')
        self.assertAlmostEqual(score,.05)
        self.assertEqual([x['concept'] for x in m],['very_good','bad'])
        self.assertEqual(f.extract('xyz')[0],0.)
        self.assertEqual(f.extract('very. good')[1][0]['concept'],'good')

    def test_group_split_and_report_overlap(self):
        rows=[dict(sentence=f'{i} test',label=i%2,idx=i) for i in range(100)]
        rows.append(dict(sentence='0 TEST',label=0,idx=100))
        train,tune=split_rows(rows,[dict(sentence='2 test')],fraction=.2)
        a={normalize(r['sentence']) for r in train}; b={normalize(r['sentence']) for r in tune}
        self.assertFalse(a & b)
        self.assertNotIn('2 test',a|b)
        self.assertTrue('0 test' in a or '0 test' in b)

    def test_last_valid_left_and_right_padding(self):
        h=torch.arange(8).reshape(2,4,1).float()
        self.assertEqual(last_valid(h,torch.tensor([[1,1,0,0],[0,1,1,1]])).flatten().tolist(),[1.,7.])
        with self.assertRaises(ValueError): last_valid(h,torch.zeros(2,4,dtype=torch.long))

    def test_grid(self):
        grid=experiment_grid(42)
        self.assertEqual(len({r['id'] for r in grid}),8)
        for m in ('flex','lora'):
            for k in (False,True):
                self.assertEqual(len([r for r in grid if r['method']==m and r['knowledge']==k]),2)

    def test_trainable_boundaries_and_backward(self):
        args,_=fixtures()
        for m in ('flex','lora'):
            model=tiny_model(args,{},dict(method=m,knowledge=True),torch.device('cpu'))
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            if m=='flex':
                self.assertTrue(any('embed_tokens' in n for n in names))
                self.assertFalse(any('layers.0.' in n for n in names))
                self.assertTrue(any('layers.1.' in n for n in names))
            else:
                self.assertTrue(any('lora_' in n for n in names))
                self.assertTrue(all('lora_' in n or n.startswith('head.') for n in names))
            y=model(torch.tensor([[1,2,3]]),torch.ones(1,3,dtype=torch.long),torch.tensor([.5]))
            y.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'head' not in n))

    def test_knowledge_off_ignores_score(self):
        args,_=fixtures()
        model=tiny_model(args,{},dict(method='flex',knowledge=False),torch.device('cpu')).eval()
        ids=torch.tensor([[1,2,3]]); mask=torch.ones_like(ids)
        self.assertTrue(torch.equal(model(ids,mask,torch.tensor([-1.])),model(ids,mask,torch.tensor([1.]))))

    def test_corrupt_checkpoint_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            save_snapshot(temp,'step-000000001',{'x':1})
            save_snapshot(temp,'step-000000002',{'x':2})
            (Path(temp)/'step-000000002.pt').write_bytes(b'corrupted')
            _,state=latest_snapshot(temp)
            self.assertEqual(state['x'],1)

    def test_resume_matches_uninterrupted_and_finalize(self):
        args,data=fixtures()
        grid=experiment_grid(42)
        protocol=dict(model_revision='offline',grid=grid)
        with tempfile.TemporaryDirectory() as temp, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('nexus_pilot.run.build_model',side_effect=tiny_model), \
             patch('nexus_pilot.run.AutoTokenizer.from_pretrained',return_value=TinyTokenizer()):
            root=Path(temp)/'resumed'; reference=Path(temp)/'reference'
            # Interrupt after first optimizer step, then reconstruct model/optimizer.
            self.assertFalse(run_one(args,root,protocol,data,grid[0],time.monotonic()-1))
            self.assertTrue(run_one(args,root,protocol,data,grid[0],float('inf')))
            self.assertTrue(run_one(args,reference,protocol,data,grid[0],float('inf')))
            a=json.loads((root/grid[0]['id']/'complete.json').read_text())
            b=json.loads((reference/grid[0]['id']/'complete.json').read_text())
            self.assertEqual(a['tune'],b['tune'])
            wa=load_snapshot(root/grid[0]['id']/a['best_checkpoint'])['model']
            wb=load_snapshot(reference/grid[0]['id']/b['best_checkpoint'])['model']
            self.assertTrue(all(torch.equal(wa[k],wb[k]) for k in wa))
            for c in grid[1:]: run_one(args,root,protocol,data,c,float('inf'))
            finalize(args,root,protocol,data)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(len(summary),4)
            self.assertTrue((root/'summary.csv').exists())
            self.assertEqual(len(list(root.glob('*/predictions.csv'))),4)
            self.assertFalse(list(root.glob('*/step-*.pt')))


if __name__=='__main__':
    unittest.main()
