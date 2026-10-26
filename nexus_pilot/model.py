import torch
from torch import nn


def last_valid(hidden, attention_mask):
    """Correct for both left and right padding; never assumes a CLS token."""
    indices = torch.arange(attention_mask.shape[1], device=hidden.device)
    last = indices.unsqueeze(0).expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if (last < 0).any():
        raise ValueError('All-padding sequence')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class PilotClassifier(nn.Module):
    def __init__(self, backbone, method, knowledge, seed=42, rank=8, alpha=16):
        super().__init__()
        self.knowledge = knowledge
        hidden = backbone.config.hidden_size
        for p in backbone.parameters():
            p.requires_grad = False
        if method == 'flex':
            for p in backbone.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in backbone.layers[-1].parameters():
                p.requires_grad = True
        elif method == 'lora':
            from peft import LoraConfig, get_peft_model
            backbone = get_peft_model(backbone, LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj'], bias='none'))
        else:
            raise ValueError(method)
        self.backbone = backbone
        # Identical head initialization and dimensionality in all four arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            self.head = nn.Linear(hidden + 1, 2)

    def forward(self, input_ids, attention_mask, symbolic_score):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state
        pooled = last_valid(h, attention_mask)
        s = symbolic_score[:, None].to(pooled.dtype)
        if not self.knowledge:
            s = torch.zeros_like(s)
        return self.head(torch.cat([pooled, s], dim=-1))

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore_trainable(self, state):
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if expected != set(state):
            raise ValueError('Checkpoint trainable parameter names do not match model')
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in state:
                    p.copy_(state[n])
