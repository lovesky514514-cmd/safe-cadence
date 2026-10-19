"""Minimal SGD TransE with actual contrastive margin-ranking loss."""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class TrainConfig:
    dim: int = 24
    steps: int = 450
    batch_size: int = 128
    negatives: int = 4
    lr: float = 0.025
    margin: float = 1.0
    seed: int = 7
    relation_anchor_strength: float = 0.0
    entity_anchor_strength: float = 0.0
    ewc_strength: float = 0.0


def train_transe(triples, n_entities, n_relations, cfg=None, init=None, forbidden_positive=None, ewc=None):
    """Return E,R,metadata. Train negatives against TRAIN facts only to avoid leakage."""
    cfg = cfg or TrainConfig()
    torch.set_num_threads(1)
    g = torch.Generator(device='cpu').manual_seed(cfg.seed)
    pos = torch.as_tensor(triples, dtype=torch.long)
    if len(pos) == 0:
        raise ValueError('Empty training set')
    if init is None:
        scale = 0.4
        ent = torch.nn.Parameter((torch.rand(n_entities, cfg.dim, generator=g)*2-1)*scale)
        rel = torch.nn.Parameter((torch.rand(n_relations, cfg.dim, generator=g)*2-1)*scale)
    else:
        ent = torch.nn.Parameter(torch.tensor(init[0], dtype=torch.float32).clone())
        rel = torch.nn.Parameter(torch.tensor(init[1], dtype=torch.float32).clone())
    if (cfg.relation_anchor_strength or cfg.entity_anchor_strength or cfg.ewc_strength) and init is None:
        raise ValueError('Relation anchor requires a pre-update model')
    relation_anchor = torch.tensor(init[1], dtype=torch.float32) if init is not None else None
    entity_anchor = torch.tensor(init[0], dtype=torch.float32) if init is not None else None
    if cfg.ewc_strength and ewc is None:
        raise ValueError("ewc_strength needs old-task diagonal importance")
    if ewc is not None:
        imp_e,imp_r=[torch.tensor(z,dtype=torch.float32) for z in ewc]
        if imp_e.shape != ent.shape or imp_r.shape != rel.shape:
            raise ValueError("EWC importance shape mismatch")
    opt = torch.optim.Adam([ent, rel], lr=cfg.lr)
    known_train = set(map(tuple, triples.tolist()))
    if forbidden_positive is not None:
        known_train.update(map(tuple, np.asarray(forbidden_positive, dtype=np.int64).tolist()))
    losses = []
    for i in range(cfg.steps):
        indices = torch.randint(len(pos), (cfg.batch_size,), generator=g)
        batch = pos[indices]
        neg = batch.repeat_interleave(cfg.negatives, dim=0).clone()
        side = torch.randint(0, 2, (len(neg),), generator=g)
        corrup = torch.randint(n_entities, (len(neg),), generator=g)
        neg[side == 0, 0] = corrup[side == 0]
        neg[side == 1, 2] = corrup[side == 1]
        # Resample known true training facts. This avoids training false negatives.
        for j in range(len(neg)):
            tries = 0
            while tuple(neg[j].tolist()) in known_train and tries < 20:
                neg[j, 0 if side[j] == 0 else 2] = torch.randint(n_entities, (1,), generator=g).item()
                tries += 1
            if tuple(neg[j].tolist()) in known_train:
                # Shift along candidate universe to guarantee another entity when possible.
                axis = 0 if side[j] == 0 else 2
                found = False
                for candidate in range(n_entities):
                    neg[j, axis] = candidate
                    if tuple(neg[j].tolist()) not in known_train:
                        found = True
                        break
                if not found:
                    raise ValueError('No false-negative-free corruption available')
        def distance(rows):
            return torch.linalg.vector_norm(ent[rows[:, 0]] + rel[rows[:, 1]] - ent[rows[:, 2]], dim=1)
        pos_dist = distance(batch).repeat_interleave(cfg.negatives)
        neg_dist = distance(neg)
        loss = F.relu(cfg.margin + pos_dist-neg_dist).mean()
        if cfg.relation_anchor_strength:
            loss = loss + 0.5 * cfg.relation_anchor_strength * (rel-relation_anchor).square().sum()
        if cfg.entity_anchor_strength:
            loss = loss + 0.5 * cfg.entity_anchor_strength * (ent-entity_anchor).square().sum()
        if cfg.ewc_strength:
            loss = loss + 0.5 * cfg.ewc_strength * (((ent-entity_anchor).square()*imp_e).sum() / n_entities + ((rel-relation_anchor).square()*imp_r).sum() / n_relations)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Norm constraint avoids trivial scale explosion.
        with torch.no_grad():
            ent.copy_(F.normalize(ent, p=2, dim=1))
        if i in (0, cfg.steps-1) or i % 100 == 0:
            losses.append([int(i), round(float(loss.item()), 6)])
    return ent.detach().numpy().astype(np.float64), rel.detach().numpy().astype(np.float64), {'loss_log': losses, 'negative_sampling': 'head/tail corrupt; training-truth rejection', 'loss': 'relu(margin + positive L2 distance - negative L2 distance)', 'relation_anchor_strength': cfg.relation_anchor_strength, 'entity_anchor_strength': cfg.entity_anchor_strength, 'ewc_strength': cfg.ewc_strength}
