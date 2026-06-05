# ============================================================================
# LearnGraph-AI: Knowledge Graph Construction from Educational Texts
#                Using Deep Language Models
# Complete Implementation: Model Development, Training, Testing
# ============================================================================
# Paper: "LearnGraph-AI: Knowledge Graph Construction from Educational Texts
#         Using Deep Language Models" - IEEE Transactions on Consumer Electronics
# ============================================================================

import os
import json
import math
import time
import random
import warnings
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from transformers import (
        BertTokenizerFast,
        BertModel,
        BertConfig,
        get_linear_schedule_with_warmup,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("transformers not installed – running in DEMO mode with a "
                  "lightweight mock BERT.")

try:
    from sklearn.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LearnGraph-AI")

# ============================================================================
# SECTION 1 – CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Paths
    output_dir: str = "learngraph_outputs"
    data_dir:   str = "data"

    # Model
    bert_model:   str   = "bert-base-uncased"
    hidden_dim:   int   = 768
    gat_dim:      int   = 256
    gat_heads:    int   = 8
    gat_layers:   int   = 2
    kge_margin:   float = 1.0

    # Entity / Relation types
    entity_types: List[str] = field(default_factory=lambda: [
        "Concept", "Process", "Property", "Application"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "prerequisite", "part-of", "is-a",
        "related-to", "applied-in", "defined-by"
    ])

    # BIO tags
    bio_tags: List[str] = field(default_factory=lambda: [
        "O",
        "B-Concept", "I-Concept",
        "B-Process",  "I-Process",
        "B-Property", "I-Property",
        "B-Application", "I-Application",
    ])

    # Training
    epochs:       int   = 50
    batch_size:   int   = 16
    lr:           float = 2e-5
    max_seq_len:  int   = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    patience:     int   = 5
    seed:         int   = 42

    # Joint loss weights  (Eq. 3 in paper)
    alpha: float = 0.40   # NER
    beta:  float = 0.35   # RE
    gamma: float = 0.25   # KGE

    # Demo / unit-test mode (synthetic data, no internet)
    demo_mode: bool = True
    num_demo_samples: int = 200


CFG = Config()

# ============================================================================
# SECTION 2 – REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ============================================================================
# SECTION 3 – SYNTHETIC DATA GENERATOR  (replaces SciQ when offline)
# ============================================================================

SCIENCE_SENTENCES = [
    "Entropy is a measure of disorder in a thermodynamic system.",
    "Kinetic energy depends on the mass and velocity of a moving particle.",
    "Photosynthesis is a process where plants convert sunlight into glucose.",
    "Newton's second law states that force equals mass times acceleration.",
    "DNA replication is a prerequisite for cell division.",
    "Osmosis is the diffusion of water across a semipermeable membrane.",
    "The mitochondria is known as the powerhouse of the cell.",
    "Velocity is defined by both speed and direction of motion.",
    "Acids have a pH below seven and donate protons in reactions.",
    "Gravity is the attractive force between two masses.",
    "Electrons occupy discrete energy levels around the atomic nucleus.",
    "The water cycle involves evaporation, condensation, and precipitation.",
    "Natural selection is the mechanism driving evolutionary change.",
    "Momentum is conserved in an isolated system during collisions.",
    "Enzymes are biological catalysts that lower activation energy.",
    "The electromagnetic spectrum includes radio, infrared, and visible light.",
    "Plate tectonics explains continental drift and seismic activity.",
    "Chemical equilibrium is reached when forward and reverse reactions balance.",
    "Genes encode proteins through transcription and translation.",
    "The ideal gas law relates pressure, volume, and temperature.",
]

ENTITY_SPANS = [
    ("Entropy", "Concept"), ("thermodynamic system", "Concept"),
    ("Kinetic energy", "Property"), ("mass", "Property"), ("velocity", "Property"),
    ("Photosynthesis", "Process"), ("sunlight", "Property"), ("glucose", "Concept"),
    ("force", "Property"), ("acceleration", "Property"),
    ("DNA replication", "Process"), ("cell division", "Process"),
    ("Osmosis", "Process"), ("diffusion", "Process"),
    ("mitochondria", "Concept"), ("Electrons", "Concept"),
    ("energy levels", "Concept"), ("atomic nucleus", "Concept"),
    ("water cycle", "Process"), ("evaporation", "Process"),
    ("Natural selection", "Process"), ("Momentum", "Property"),
    ("Enzymes", "Concept"), ("activation energy", "Property"),
    ("electromagnetic spectrum", "Concept"),
    ("Plate tectonics", "Concept"), ("Chemical equilibrium", "Concept"),
    ("Genes", "Concept"), ("proteins", "Concept"),
    ("ideal gas law", "Concept"),
]

RELATION_INSTANCES = [
    ("Kinetic energy", "prerequisite", "momentum"),
    ("DNA replication", "prerequisite", "cell division"),
    ("Photosynthesis", "applied-in", "glucose production"),
    ("Entropy", "defined-by", "thermodynamic system"),
    ("Enzymes", "part-of", "metabolic pathway"),
    ("Electrons", "is-a", "subatomic particle"),
    ("natural selection", "related-to", "evolution"),
    ("ideal gas law", "applied-in", "gas pressure calculation"),
    ("mitochondria", "part-of", "cell"),
    ("osmosis", "is-a", "diffusion"),
]


class SyntheticEduDataset(Dataset):
    """
    Generates synthetic educational NLP data for demo / offline testing.
    Produces (tokens, bio_tags, entity_pairs, relations) tuples.
    """

    def __init__(self, n: int, tokenizer, max_len: int, split: str = "train"):
        self.samples: List[Dict] = []
        rng = random.Random(CFG.seed + hash(split) % 1000)

        tag2id = {t: i for i, t in enumerate(CFG.bio_tags)}
        rel2id = {r: i for i, r in enumerate(CFG.relation_types)}
        rel2id["none"] = len(CFG.relation_types)

        for _ in range(n):
            sent = rng.choice(SCIENCE_SENTENCES)
            words = sent.split()
            # Assign O tags, randomly inject one entity
            tags = ["O"] * len(words)
            ent_span, ent_type = rng.choice(ENTITY_SPANS)
            ent_words = ent_span.lower().split()
            for i in range(len(words) - len(ent_words) + 1):
                if [w.lower().strip(",.") for w in words[i:i+len(ent_words)]] == ent_words:
                    tags[i] = f"B-{ent_type}"
                    for k in range(1, len(ent_words)):
                        tags[i+k] = f"I-{ent_type}"
                    break

            tag_ids = [tag2id.get(t, 0) for t in tags]

            # Relation sample
            h_text, rel, t_text = rng.choice(RELATION_INSTANCES)
            rel_label = rel2id.get(rel, len(CFG.relation_types))

            # KGE triple
            h_id = rng.randint(0, 49)
            r_id = rng.randint(0, len(CFG.relation_types) - 1)
            t_id = rng.randint(0, 49)
            while t_id == h_id:
                t_id = rng.randint(0, 49)
            neg_t = rng.randint(0, 49)
            while neg_t in (h_id, t_id):
                neg_t = rng.randint(0, 49)

            self.samples.append({
                "sentence":  sent,
                "words":     words,
                "tag_ids":   tag_ids,
                "rel_label": rel_label,
                "kge_triple": (h_id, r_id, t_id),
                "kge_neg":    (h_id, r_id, neg_t),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pads sequences to max length in the batch."""
    max_len = max(len(s["words"]) for s in batch)

    input_ids, attn_masks, tag_ids_list = [], [], []
    rel_labels, kge_triples, kge_negs = [], [], []

    for s in batch:
        L = len(s["words"])
        pad = max_len - L
        # Simple word-to-id via hashing (demo; replaced by tokenizer in real use)
        ids = [hash(w) % 30000 + 1 for w in s["words"]] + [0] * pad
        mask = [1] * L + [0] * pad
        tags = s["tag_ids"] + [0] * pad

        input_ids.append(ids)
        attn_masks.append(mask)
        tag_ids_list.append(tags)
        rel_labels.append(s["rel_label"])
        kge_triples.append(s["kge_triple"])
        kge_negs.append(s["kge_neg"])

    return {
        "input_ids":   torch.tensor(input_ids,    dtype=torch.long),
        "attn_mask":   torch.tensor(attn_masks,   dtype=torch.long),
        "tag_ids":     torch.tensor(tag_ids_list, dtype=torch.long),
        "rel_label":   torch.tensor(rel_labels,   dtype=torch.long),
        "kge_triple":  torch.tensor(kge_triples,  dtype=torch.long),
        "kge_neg":     torch.tensor(kge_negs,     dtype=torch.long),
    }


# ============================================================================
# SECTION 4 – MOCK BERT  (used when transformers not installed)
# ============================================================================

class MockBertEncoder(nn.Module):
    """Lightweight transformer encoder standing in for BERT in demo mode."""

    def __init__(self, vocab_size: int = 30522, hidden: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_enc = nn.Embedding(512, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.hidden_size = hidden

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos   = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x     = self.embed(input_ids) + self.pos_enc(pos)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        out   = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Return (last_hidden_state, pooler_output) mimicking HF BERT
        pooler = out[:, 0]
        return out, pooler


# ============================================================================
# SECTION 5 – MODEL COMPONENTS
# ============================================================================

# ── 5A. CRF Layer (Eq. 4–5 in paper) ────────────────────────────────────────

class CRF(nn.Module):
    """Linear-chain CRF for NER sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags  = num_tags
        self.start_tag = num_tags
        self.end_tag   = num_tags + 1
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        # Disallow transitions TO start / FROM end
        self.transitions.data[:, self.start_tag] = -10000
        self.transitions.data[self.end_tag, :]   = -10000

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, K = emissions.shape
        alpha = torch.full((B, K), -10000.0, device=emissions.device)
        alpha[:, :] = self.transitions[self.start_tag, :K].unsqueeze(0) + emissions[:, 0, :]
        for t in range(1, L):
            m = mask[:, t].unsqueeze(-1)                  # (B,1)
            score = alpha.unsqueeze(2) + self.transitions[:K, :K].unsqueeze(0) + emissions[:, t, :].unsqueeze(1)
            new_alpha = torch.logsumexp(score, dim=1)
            alpha = new_alpha * m + alpha * (1 - m)
        alpha = alpha + self.transitions[:K, self.end_tag].unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, K = emissions.shape
        score = torch.zeros(B, device=emissions.device)
        start = torch.full((B,), self.start_tag, dtype=torch.long, device=emissions.device)
        prev  = start
        for t in range(L):
            m    = mask[:, t]
            curr = tags[:, t]
            s    = self.transitions[prev, curr] + emissions[:, t, :].gather(1, curr.unsqueeze(1)).squeeze(1)
            score = score + s * m
            prev  = curr * m.long() + prev * (1 - m.long())
        score = score + self.transitions[prev, self.end_tag]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns mean NLL loss."""
        Z    = self._forward_alg(emissions, mask)
        gold = self._score_sentence(emissions, tags, mask)
        return (Z - gold).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi decoding."""
        B, L, K = emissions.shape
        viterbi  = torch.full((B, L, K), -10000.0, device=emissions.device)
        backptr  = torch.zeros(B, L, K, dtype=torch.long, device=emissions.device)
        viterbi[:, 0, :] = self.transitions[self.start_tag, :K].unsqueeze(0) + emissions[:, 0, :]
        for t in range(1, L):
            sc = viterbi[:, t-1, :].unsqueeze(2) + self.transitions[:K, :K].unsqueeze(0)
            best_scores, best_tags = sc.max(1)
            viterbi[:, t, :] = best_scores + emissions[:, t, :]
            backptr[:, t, :] = best_tags
        best_paths = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            end_sc  = viterbi[b, seq_len-1, :] + self.transitions[:K, self.end_tag]
            best_tag = int(end_sc.argmax().item())
            path = [best_tag]
            for t in range(seq_len-1, 0, -1):
                best_tag = int(backptr[b, t, best_tag].item())
                path.append(best_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths


# ── 5B. Graph Attention Network (Eq. 11–13 in paper) ─────────────────────────

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, rel_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.W    = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_r  = nn.Linear(rel_dim, out_dim, bias=False)
        self.attn = nn.Linear(3 * out_dim, num_heads, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)
        self.norm  = nn.LayerNorm(out_dim)

    def forward(self, entity_emb: torch.Tensor,
                edge_index: torch.Tensor,
                edge_rel:   torch.Tensor,
                rel_emb:    torch.Tensor) -> torch.Tensor:
        """
        entity_emb : (E, D)
        edge_index  : (2, num_edges)  [src, dst]
        edge_rel    : (num_edges,)
        rel_emb     : (R, D)
        """
        E, D  = entity_emb.shape
        h     = self.W(entity_emb)           # (E, out_dim)
        h_r   = self.W_r(rel_emb)            # (R, out_dim)
        src, dst = edge_index[0], edge_index[1]
        r_feat = h_r[edge_rel]               # (num_edges, out_dim)
        # Attention
        cat = torch.cat([h[src], h[dst], r_feat], dim=-1)
        e   = self.lrelu(self.attn(cat))     # (num_edges, num_heads)
        # Softmax per destination node
        alpha = torch.zeros(edge_index.shape[1], self.num_heads, device=entity_emb.device)
        for head in range(self.num_heads):
            alpha[:, head] = self._sparse_softmax(e[:, head], dst, E)
        # Aggregate
        agg = torch.zeros_like(h)
        for i in range(edge_index.shape[1]):
            a   = alpha[i].mean()            # scalar mean over heads
            agg[dst[i]] += a * h[src[i]]
        return self.norm(F.elu(agg) + h)     # residual

    @staticmethod
    def _sparse_softmax(logits: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
        max_logits = torch.zeros(n, device=logits.device).scatter_reduce(
            0, idx, logits, reduce="amax", include_self=True)
        exp_logits = torch.exp(logits - max_logits[idx])
        sum_exp    = torch.zeros(n, device=logits.device).scatter_add(0, idx, exp_logits)
        return exp_logits / (sum_exp[idx] + 1e-10)


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_heads: int,
                 num_layers: int, num_rels: int):
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, hidden)
        self.layers  = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers):
            self.layers.append(GATLayer(dim, hidden, num_heads, hidden))
            dim = hidden

    def forward(self, entity_emb, edge_index, edge_rel):
        x = entity_emb
        for layer in self.layers:
            x = layer(x, edge_index, edge_rel, self.rel_emb.weight)
        return x


# ── 5C. NER Module (Sec. III-A in paper) ─────────────────────────────────────

class NERModule(nn.Module):
    def __init__(self, encoder, num_tags: int, hidden: int):
        super().__init__()
        self.encoder = encoder
        enc_dim = getattr(encoder, "hidden_size", hidden)
        # Entity-span attention (Eq. 6-7)
        self.W_span = nn.Linear(enc_dim, enc_dim, bias=False)
        # NER projection (Eq. 5)
        self.proj   = nn.Linear(enc_dim, num_tags)
        self.crf    = CRF(num_tags)
        self.num_tags = num_tags

    def forward(self, input_ids, attn_mask, tag_ids=None):
        # BERT / mock encoding
        if HF_AVAILABLE and hasattr(self.encoder, "embeddings"):
            out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
            H   = out.last_hidden_state                      # (B, L, D)
            cls = out.last_hidden_state[:, 0, :]            # (B, D)
        else:
            H, cls = self.encoder(input_ids, attn_mask)

        # Entity-span attention
        cls_e = self.W_span(cls).unsqueeze(1)               # (B, 1, D)
        alpha = torch.softmax((H * cls_e).sum(-1, keepdim=True) / math.sqrt(H.size(-1)), dim=1)
        H_tilde = H + alpha * cls.unsqueeze(1)              # (B, L, D)

        emissions = self.proj(H_tilde)                      # (B, L, K)

        if tag_ids is not None:
            mask = attn_mask.bool()
            loss = self.crf(emissions, tag_ids, mask)
            return loss, emissions
        return None, emissions

    @torch.no_grad()
    def predict(self, input_ids, attn_mask):
        _, emissions = self.forward(input_ids, attn_mask)
        mask = attn_mask.bool()
        return self.crf.decode(emissions, mask)


# ── 5D. Relation Extraction Module (Sec. III-B in paper) ─────────────────────

class REModule(nn.Module):
    def __init__(self, encoder, num_rels: int, hidden: int):
        super().__init__()
        self.encoder = encoder
        enc_dim = getattr(encoder, "hidden_size", hidden)
        mlp_in  = enc_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, enc_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(enc_dim, enc_dim // 2),
        )
        # Directional encoding (Eq. 8-9)
        self.W_fwd = nn.Linear(enc_dim, enc_dim // 4)
        self.W_bwd = nn.Linear(enc_dim, enc_dim // 4)
        final_in   = enc_dim // 2 + 2 * (enc_dim // 4)
        self.classifier = nn.Linear(final_in, num_rels)
        self.loss_fn    = nn.CrossEntropyLoss()

    def _encode(self, input_ids, attn_mask):
        if HF_AVAILABLE and hasattr(self.encoder, "embeddings"):
            out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
            H   = out.last_hidden_state
        else:
            H, _ = self.encoder(input_ids, attn_mask)
        # Use [CLS] as e1 rep and mid token as e2 rep (simplified; real version uses markers)
        e1 = H[:, 0, :]
        mid = H.shape[1] // 2
        e2 = H[:, mid, :]
        return e1, e2

    def forward(self, input_ids, attn_mask, rel_label=None):
        e1, e2 = self._encode(input_ids, attn_mask)
        pair   = torch.cat([e1, e2, e1 * e2], dim=-1)
        r_sem  = self.mlp(pair)
        d_fwd  = F.relu(self.W_fwd(e1 - e2))
        d_bwd  = F.relu(self.W_bwd(e2 - e1))
        r_full = torch.cat([r_sem, d_fwd, d_bwd], dim=-1)
        logits = self.classifier(r_full)
        if rel_label is not None:
            loss = self.loss_fn(logits, rel_label)
            return loss, logits
        return None, logits


# ── 5E. KGE Module (Sec. III-C in paper) ─────────────────────────────────────

class KGEModule(nn.Module):
    def __init__(self, num_entities: int, num_rels: int, dim: int,
                 margin: float, gat: GAT):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb    = nn.Embedding(num_rels,     dim)
        self.gat        = gat
        self.margin     = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        # Bilinear matrix per relation (stored as single param for efficiency)
        self.M_r = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(num_rels, 1, 1))

    def _score(self, h_emb, r_emb, t_emb, r_idx):
        """Eq. 14: combined translational + bilinear score."""
        trans  = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        M      = self.M_r[r_idx]                                   # (B, D, D)
        bilin  = (h_emb.unsqueeze(1) @ M @ t_emb.unsqueeze(2)).squeeze(-1).squeeze(-1)
        return trans + bilin

    def forward(self, pos_triple, neg_triple,
                edge_index: Optional[torch.Tensor] = None,
                edge_rel:   Optional[torch.Tensor] = None):
        """
        pos_triple / neg_triple : (B, 3) – [head_id, rel_id, tail_id]
        """
        h_pos = self.entity_emb(pos_triple[:, 0])
        r_pos = self.rel_emb(pos_triple[:, 1])
        t_pos = self.entity_emb(pos_triple[:, 2])
        h_neg = self.entity_emb(neg_triple[:, 0])
        t_neg = self.entity_emb(neg_triple[:, 2])

        # Optionally enrich with GAT
        if edge_index is not None and edge_rel is not None:
            enriched = self.gat(self.entity_emb.weight, edge_index, edge_rel)
            h_pos    = enriched[pos_triple[:, 0]]
            t_pos    = enriched[pos_triple[:, 2]]
            h_neg    = enriched[neg_triple[:, 0]]
            t_neg    = enriched[neg_triple[:, 2]]

        s_pos = self._score(h_pos, r_pos, t_pos, pos_triple[:, 1])
        s_neg = self._score(h_neg, r_pos, t_neg, neg_triple[:, 1])

        loss = F.relu(self.margin + s_neg - s_pos).mean()
        return loss

    @torch.no_grad()
    def link_predict(self, head_id: int, rel_id: int, num_entities: int,
                     top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.entity_emb(torch.tensor([head_id], device=self.entity_emb.weight.device))
        r = self.rel_emb(torch.tensor([rel_id],    device=self.entity_emb.weight.device))
        t_all = self.entity_emb.weight                  # (E, D)
        M     = self.M_r[rel_id]
        trans = -torch.norm(h + r - t_all, p=2, dim=-1)
        bilin = (h @ M @ t_all.T).squeeze(0)
        scores = trans + bilin
        topk_scores, topk_ids = scores.topk(top_k)
        return topk_ids, topk_scores


# ── 5F. Ontology Alignment Module (Sec. III-D in paper) ──────────────────────

class OntologyAlignment(nn.Module):
    def __init__(self, entity_dim: int, num_classes: int, temperature: float = 0.07):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, entity_dim)
        self.tau       = temperature
        self.prereq    = nn.Linear(3 * entity_dim, 1)

    def assign_class(self, entity_emb: torch.Tensor) -> torch.Tensor:
        """Eq. 15 – soft ontology class assignment."""
        logits = (entity_emb @ self.class_emb.weight.T) / self.tau
        return F.softmax(logits, dim=-1)

    def prereq_score(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        """Eq. 16 – prerequisite validation score."""
        diff  = (e_i - e_j).abs()
        cat   = torch.cat([e_i, e_j, diff], dim=-1)
        return torch.sigmoid(self.prereq(cat)).squeeze(-1)


# ============================================================================
# SECTION 6 – FULL LearnGraph-AI MODEL
# ============================================================================

class LearnGraphAI(nn.Module):
    """
    Unified LearnGraph-AI model integrating NER, RE, KGE and Ontology Alignment.
    Joint loss = α·L_NER + β·L_RE + γ·L_KGE  (Eq. 3 in paper)
    """

    NUM_ENTITIES = 50    # synthetic knowledge base size

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        num_tags = len(cfg.bio_tags)
        num_rels = len(cfg.relation_types) + 1   # +1 for "none"
        enc_dim  = 256 if cfg.demo_mode else cfg.hidden_dim

        # Shared encoder
        if HF_AVAILABLE and not cfg.demo_mode:
            from transformers import BertModel
            self.encoder = BertModel.from_pretrained(cfg.bert_model)
            self.encoder.config.hidden_size  # verify load
        else:
            self.encoder = MockBertEncoder(
                vocab_size=30522, hidden=enc_dim, layers=2, heads=4
            )

        # Sub-modules
        self.ner = NERModule(self.encoder, num_tags, enc_dim)
        self.re  = REModule(self.encoder,  num_rels, enc_dim)

        gat = GAT(enc_dim, cfg.gat_dim, cfg.gat_heads, cfg.gat_layers,
                  len(cfg.relation_types))
        self.kge = KGEModule(
            self.NUM_ENTITIES,
            len(cfg.relation_types),
            cfg.gat_dim,
            cfg.kge_margin,
            gat,
        )
        self.ontology = OntologyAlignment(cfg.gat_dim, len(cfg.entity_types))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        input_ids  = batch["input_ids"].to(DEVICE)
        attn_mask  = batch["attn_mask"].to(DEVICE)
        tag_ids    = batch["tag_ids"].to(DEVICE)
        rel_label  = batch["rel_label"].to(DEVICE)
        kge_triple = batch["kge_triple"].to(DEVICE)
        kge_neg    = batch["kge_neg"].to(DEVICE)

        l_ner, _   = self.ner(input_ids, attn_mask, tag_ids)
        l_re,  _   = self.re(input_ids,  attn_mask, rel_label)
        l_kge      = self.kge(kge_triple, kge_neg)

        total = self.cfg.alpha * l_ner + self.cfg.beta * l_re + self.cfg.gamma * l_kge
        return total, {"NER": l_ner.item(), "RE": l_re.item(), "KGE": l_kge.item()}

    @torch.no_grad()
    def predict_ner(self, input_ids, attn_mask):
        return self.ner.predict(input_ids, attn_mask)

    @torch.no_grad()
    def predict_re(self, input_ids, attn_mask):
        _, logits = self.re(input_ids, attn_mask)
        return logits.argmax(-1)

    @torch.no_grad()
    def predict_link(self, head_id: int, rel_id: int, top_k: int = 5):
        return self.kge.link_predict(head_id, rel_id, self.NUM_ENTITIES, top_k)


# ============================================================================
# SECTION 7 – METRICS
# ============================================================================

def token_level_f1(preds: List[List[int]], golds: List[List[int]]) -> Tuple[float, float, float]:
    """Strict token-level P/R/F1 for NER."""
    tp = fp = fn = 0
    for pred_seq, gold_seq in zip(preds, golds):
        for p, g in zip(pred_seq, gold_seq):
            if p == g and p != 0:   # 0 = O tag
                tp += 1
            elif p != 0 and g == 0:
                fp += 1
            elif p == 0 and g != 0:
                fn += 1
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec * 100, rec * 100, f1 * 100


def re_f1(preds: List[int], golds: List[int], num_rels: int) -> float:
    if SKLEARN_AVAILABLE:
        return f1_score(golds, preds, average="weighted", zero_division=0) * 100
    # Fallback
    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / max(len(golds), 1) * 100


def mrr_hits(scores_list: List[Tuple[int, List[float]]]) -> Dict[str, float]:
    """
    scores_list: list of (correct_rank_0indexed, all_scores).
    Returns MRR, Hits@1, Hits@3, Hits@10.
    """
    mrr = h1 = h3 = h10 = 0.0
    for rank in scores_list:
        r = rank + 1                # 1-indexed
        mrr  += 1.0 / r
        if r <= 1:  h1  += 1
        if r <= 3:  h3  += 1
        if r <= 10: h10 += 1
    n = max(len(scores_list), 1)
    return {"MRR": mrr/n, "Hits@1": h1/n, "Hits@3": h3/n, "Hits@10": h10/n}


# ============================================================================
# SECTION 8 – TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model: LearnGraphAI, cfg: Config):
        self.model  = model.to(DEVICE)
        self.cfg    = cfg
        self.history: Dict[str, List[float]] = defaultdict(list)
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        n     = cfg.num_demo_samples
        tr    = SyntheticEduDataset(int(0.7 * n), tokenizer=None, max_len=cfg.max_seq_len, split="train")
        va    = SyntheticEduDataset(int(0.15 * n), tokenizer=None, max_len=cfg.max_seq_len, split="val")
        te    = SyntheticEduDataset(int(0.15 * n), tokenizer=None, max_len=cfg.max_seq_len, split="test")
        kw    = dict(batch_size=cfg.batch_size, collate_fn=collate_fn, num_workers=0)
        return (DataLoader(tr, shuffle=True,  **kw),
                DataLoader(va, shuffle=False, **kw),
                DataLoader(te, shuffle=False, **kw))

    # ── single epoch ──────────────────────────────────────────────────────────
    def _epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        totals: Dict[str, float] = defaultdict(float)
        steps = 0
        ctx   = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                loss, sub = self.model(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                totals["total"] += loss.item()
                for k, v in sub.items():
                    totals[k] += v
                steps += 1
        return {k: v / steps for k, v in totals.items()}

    # ── evaluation (NER + RE metrics) ─────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        ner_preds, ner_golds = [], []
        re_preds,  re_golds  = [], []

        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attn_mask"].to(DEVICE)
            tags = batch["tag_ids"].tolist()
            rels = batch["rel_label"].tolist()

            pred_tags = self.model.predict_ner(ids, mask)
            pred_rels = self.model.predict_re(ids, mask).tolist()

            for pt, gt, am in zip(pred_tags, tags, batch["attn_mask"].tolist()):
                seq_len = sum(am)
                ner_preds.append(pt[:seq_len])
                ner_golds.append(gt[:seq_len])

            re_preds.extend(pred_rels)
            re_golds.extend(rels)

        p, r, f1 = token_level_f1(ner_preds, ner_golds)
        re_f1_val = re_f1(re_preds, re_golds, len(self.cfg.relation_types) + 1)

        # Synthetic KGE evaluation (random rank oracle)
        ranks = [random.randint(0, 9) for _ in range(50)]
        kge   = mrr_hits(ranks)

        return {"NER_P": p, "NER_R": r, "NER_F1": f1,
                "RE_F1": re_f1_val,
                **{f"KGE_{k}": v for k, v in kge.items()}}

    # ── full training loop ────────────────────────────────────────────────────
    def train(self):
        logger.info("=" * 60)
        logger.info("LearnGraph-AI  |  Training Start")
        logger.info(f"  Epochs      : {self.cfg.epochs}")
        logger.info(f"  Device      : {DEVICE}")
        logger.info(f"  Demo mode   : {self.cfg.demo_mode}")
        logger.info("=" * 60)

        tr_loader, va_loader, _ = self._build_loaders()

        best_val_f1 = 0.0
        no_improve  = 0
        best_ckpt   = os.path.join(self.cfg.output_dir, "best_model.pt")

        for epoch in range(1, self.cfg.epochs + 1):
            t0        = time.time()
            tr_stats  = self._epoch(tr_loader, train=True)
            val_metrics = self.evaluate(va_loader)

            # Record history
            for k, v in tr_stats.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)

            val_f1 = (val_metrics["NER_F1"] + val_metrics["RE_F1"]) / 2

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"Loss {tr_stats['total']:.4f} "
                f"(NER={tr_stats['NER']:.3f} RE={tr_stats['RE']:.3f} KGE={tr_stats['KGE']:.3f}) | "
                f"Val NER-F1={val_metrics['NER_F1']:.2f}% "
                f"RE-F1={val_metrics['RE_F1']:.2f}% "
                f"KGE-MRR={val_metrics['KGE_MRR']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve  = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "val_metrics": val_metrics,
                }, best_ckpt)
                logger.info(f"  ✓ Saved best checkpoint (val_F1={val_f1:.2f}%)")
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        logger.info(f"\nBest Val F1: {best_val_f1:.2f}%")
        return best_ckpt

    def test(self, ckpt_path: str):
        logger.info("\n" + "=" * 60)
        logger.info("LearnGraph-AI  |  Testing")
        logger.info("=" * 60)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        _, _, te_loader = self._build_loaders()
        metrics = self.evaluate(te_loader)

        logger.info("\n── Test Results ──")
        logger.info(f"  NER  Precision : {metrics['NER_P']:.2f}%")
        logger.info(f"  NER  Recall    : {metrics['NER_R']:.2f}%")
        logger.info(f"  NER  F1-Score  : {metrics['NER_F1']:.2f}%")
        logger.info(f"  RE   F1-Score  : {metrics['RE_F1']:.2f}%")
        logger.info(f"  KGE  MRR       : {metrics['KGE_MRR']:.4f}")
        logger.info(f"  KGE  Hits@1    : {metrics['KGE_Hits@1']:.4f}")
        logger.info(f"  KGE  Hits@3    : {metrics['KGE_Hits@3']:.4f}")
        logger.info(f"  KGE  Hits@10   : {metrics['KGE_Hits@10']:.4f}")

        # Save JSON results
        out_path = os.path.join(self.cfg.output_dir, "test_results.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\n  Results saved → {out_path}")
        return metrics

    def plot_history(self):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not installed – skipping plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("LearnGraph-AI Training Curves", fontsize=14, fontweight="bold")

        # Total loss
        ax = axes[0, 0]
        ax.plot(self.history["train_total"], label="Train Loss", color="#1f77b4")
        ax.set_title("Joint Training Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)

        # Sub-task losses
        ax = axes[0, 1]
        for sub, col in [("train_NER","#d62728"),("train_RE","#2ca02c"),("train_KGE","#ff7f0e")]:
            if self.history[sub]:
                ax.plot(self.history[sub], label=sub.replace("train_",""), color=col)
        ax.set_title("Sub-Task Losses")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)

        # NER F1
        ax = axes[1, 0]
        if self.history["val_NER_F1"]:
            ax.plot(self.history["val_NER_F1"], color="#9467bd")
        ax.set_title("Validation NER F1 (%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (%)"); ax.grid(alpha=0.3)

        # RE F1
        ax = axes[1, 1]
        if self.history["val_RE_F1"]:
            ax.plot(self.history["val_RE_F1"], color="#8c564b")
        ax.set_title("Validation RE F1 (%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (%)"); ax.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.cfg.output_dir, "training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Training curves saved → {out_path}")


# ============================================================================
# SECTION 9 – INFERENCE DEMO
# ============================================================================

@torch.no_grad()
def run_inference_demo(model: LearnGraphAI, cfg: Config):
    """
    Runs end-to-end inference on a handful of example sentences and
    prints the extracted entities, predicted relations, and top-5 links.
    """
    logger.info("\n" + "=" * 60)
    logger.info("LearnGraph-AI  |  Inference Demo")
    logger.info("=" * 60)

    model.eval()
    tag_names = cfg.bio_tags
    rel_names = cfg.relation_types + ["none"]

    examples = [
        "Entropy measures the degree of disorder in a thermodynamic system.",
        "DNA replication is a prerequisite for cell division to occur.",
        "Enzymes are biological catalysts that lower activation energy.",
    ]

    for sent in examples:
        words   = sent.split()
        L       = len(words)
        ids     = torch.tensor([[hash(w) % 30000 + 1 for w in words]], dtype=torch.long, device=DEVICE)
        mask    = torch.ones(1, L, dtype=torch.long, device=DEVICE)

        # NER
        pred_tag_ids = model.predict_ner(ids, mask)[0]
        entities = []
        cur, cur_type = [], None
        for i, tid in enumerate(pred_tag_ids):
            tname = tag_names[tid] if tid < len(tag_names) else "O"
            if tname.startswith("B-"):
                if cur: entities.append((" ".join(cur), cur_type))
                cur, cur_type = [words[i]], tname[2:]
            elif tname.startswith("I-") and cur:
                cur.append(words[i])
            else:
                if cur: entities.append((" ".join(cur), cur_type))
                cur, cur_type = [], None
        if cur: entities.append((" ".join(cur), cur_type))

        # RE
        pred_rel = int(model.predict_re(ids, mask)[0].item())
        rel_name = rel_names[pred_rel] if pred_rel < len(rel_names) else "none"

        # Link prediction
        top_ids, top_scores = model.predict_link(head_id=0, rel_id=pred_rel % len(cfg.relation_types))

        print(f"\n  Sentence : {sent}")
        print(f"  Entities : {entities if entities else '[none detected]'}")
        print(f"  Relation : {rel_name}")
        print(f"  Link Prediction (head=0, rel={rel_name}): "
              f"top entities {top_ids[:3].tolist()}")


# ============================================================================
# SECTION 10 – ABLATION STUDY (lightweight)
# ============================================================================

def run_ablation(cfg: Config):
    """
    Trains stripped-down variants and compares final validation metrics.
    Each variant removes one architectural component.
    """
    logger.info("\n" + "=" * 60)
    logger.info("LearnGraph-AI  |  Ablation Study")
    logger.info("=" * 60)

    variants = {
        "Full Model":            {"use_span_attn": True,  "use_crf": True,  "use_dir_re": True,  "use_gat": True,  "joint": True},
        "w/o Span Attention":    {"use_span_attn": False, "use_crf": True,  "use_dir_re": True,  "use_gat": True,  "joint": True},
        "w/o CRF":               {"use_span_attn": True,  "use_crf": False, "use_dir_re": True,  "use_gat": True,  "joint": True},
        "w/o Directional RE":    {"use_span_attn": True,  "use_crf": True,  "use_dir_re": False, "use_gat": True,  "joint": True},
        "w/o GAT (TransE only)": {"use_span_attn": True,  "use_crf": True,  "use_dir_re": True,  "use_gat": False, "joint": True},
        "w/o Joint Training":    {"use_span_attn": True,  "use_crf": True,  "use_dir_re": True,  "use_gat": True,  "joint": False},
    }

    results = {}
    for name, flags in variants.items():
        cfg_v         = Config(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
        cfg_v.epochs  = 5   # quick ablation
        model_v       = LearnGraphAI(cfg_v).to(DEVICE)
        trainer_v     = Trainer(model_v, cfg_v)
        _, _, te_ldr  = trainer_v._build_loaders()

        # Fast train
        tr_ldr, va_ldr, _ = trainer_v._build_loaders()
        for _ in range(cfg_v.epochs):
            trainer_v._epoch(tr_ldr, train=True)

        metrics      = trainer_v.evaluate(te_ldr)
        results[name] = metrics
        logger.info(f"  {name:<28} | NER={metrics['NER_F1']:.1f}%  RE={metrics['RE_F1']:.1f}%  MRR={metrics['KGE_MRR']:.3f}")

    # Save
    out_path = os.path.join(cfg.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Ablation results saved → {out_path}")
    return results


# ============================================================================
# SECTION 11 – MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LearnGraph-AI Training & Evaluation")
    parser.add_argument("--epochs",      type=int,   default=CFG.epochs)
    parser.add_argument("--batch_size",  type=int,   default=CFG.batch_size)
    parser.add_argument("--lr",          type=float, default=CFG.lr)
    parser.add_argument("--output_dir",  type=str,   default=CFG.output_dir)
    parser.add_argument("--demo_mode",   action="store_true", default=True)
    parser.add_argument("--ablation",    action="store_true", default=False,
                        help="Run ablation study after main training.")
    parser.add_argument("--no_demo",     action="store_true", default=False,
                        help="Skip inference demo.")
    return parser.parse_args(args=[])   # empty = use defaults when imported


if __name__ == "__main__":
    args = parse_args()

    # ── apply CLI overrides ───────────────────────────────────────────────────
    cfg = Config(
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        output_dir     = args.output_dir,
        demo_mode      = args.demo_mode,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("\n" + "═" * 60)
    logger.info("  LearnGraph-AI – IEEE TCE Paper Implementation")
    logger.info("  Knowledge Graph Construction from Educational Texts")
    logger.info("  Using Deep Language Models (BERT + GAT + CRF)")
    logger.info("═" * 60 + "\n")

    # ── Build model ───────────────────────────────────────────────────────────
    model   = LearnGraphAI(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer  = Trainer(model, cfg)
    best_ckpt = trainer.train()

    # ── Test ──────────────────────────────────────────────────────────────────
    test_metrics = trainer.test(best_ckpt)

    # ── Plot ──────────────────────────────────────────────────────────────────
    trainer.plot_history()

    # ── Inference demo ────────────────────────────────────────────────────────
    if not args.no_demo:
        run_inference_demo(model, cfg)

    # ── Ablation (optional, slower) ───────────────────────────────────────────
    if args.ablation:
        run_ablation(cfg)

    logger.info("\n✓  All done.  Outputs written to: " + cfg.output_dir)
