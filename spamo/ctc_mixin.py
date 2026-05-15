"""
spamo/ctc_mixin.py
──────────────────
A self-contained mixin that adds a TxtCTC auxiliary loss to any SpaMo model.

Usage
─────
    class FlanT5SLT(AbstractSLT, CTCMixin):
        def __init__(self, ..., use_ctc=False, ctc_weight=0.3, ctc_blank_id=-1, ...):
            ...
            self.init_ctc(
                hidden_size  = self.t5_model.config.hidden_size,
                vocab_size   = self.t5_model.config.vocab_size,
                blank_id_cfg = ctc_blank_id,
                tokenizer    = self.t5_tokenizer,
                ctc_weight   = ctc_weight,
                use_ctc      = use_ctc,
            )

        def shared_step(self, inputs, split, batch_idx):
            ...
            visual_outputs = self.fusion_proj(visual_outputs)
            ...
            # inside the main SLT training branch, after t5_loss is computed:
            if split == 'train':
                ctc_loss = self.compute_ctc_loss(visual_outputs, visual_masks, inputs)
                loss = loss + ctc_loss          # weight already baked in
                log_dict[f"{split}/ctc_loss"] = ctc_loss

Design notes
────────────
•   init_ctc()         — call once from prepare_models(), after tokenizer exists.
•   compute_ctc_loss() — call from shared_step() in the main SLT training
                         branch only (after warmup, split == 'train').
•   The mixin owns: ctc_head (nn.Linear), ctc_criterion (nn.CTCLoss),
                    use_ctc (bool), ctc_weight (float), ctc_blank_id (int).
•   No other file is modified by this mixin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CTCMixin:
    """
    Mixin that provides a TxtCTC auxiliary loss for sign-language-to-text
    alignment.

    Architectural position
    ──────────────────────
    visual_outputs  [B, T, hidden_size]   ← output of fusion_proj, same tensor
        │                                   fed into the LLM encoder
        └─► ctc_head  Linear(H → vocab)
                │
                └─► log_softmax  →  nn.CTCLoss  →  L_TxtCTC

    The ctc_head forward pass is done in float32 regardless of the model's
    global dtype (bf16/fp16) because log_softmax + CTC over a large vocabulary
    (≈32 k tokens) is numerically unstable in reduced precision.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def init_ctc(
        self,
        hidden_size: int,
        vocab_size: int,
        blank_id_cfg: int,
        tokenizer,
        ctc_weight: float,
        use_ctc: bool,
    ) -> None:
        """
        Build the CTC head and loss criterion.

        Call this from prepare_models(), *after* the tokenizer has been loaded.

        Args:
            hidden_size:  Dimension of visual_outputs (= LLM hidden size).
            vocab_size:   Number of tokens in the text vocabulary.
            blank_id_cfg: Raw value from config. -1 means "auto from tokenizer".
            tokenizer:    The loaded text tokenizer (used to resolve blank id).
            ctc_weight:   λ_Txt — scalar weight for L_TxtCTC in total loss.
            use_ctc:      Master switch; if False the head is not created.
        """
        self.use_ctc    = use_ctc
        self.ctc_weight = ctc_weight

        if not use_ctc:
            self.ctc_blank_id  = None
            self.ctc_head      = None
            self.ctc_criterion = None
            return

        # Resolve blank token id
        # T5's pad_token_id (0) is a safe choice: it never appears in real
        # target sequences and reuses an existing vocab slot.
        self.ctc_blank_id = (
            tokenizer.pad_token_id if blank_id_cfg == -1 else blank_id_cfg
        )

        # Linear projection: visual hidden space → vocabulary logits
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

        # zero_infinity=True: samples where input_len < target_len get loss=0
        # instead of NaN/inf, which would corrupt the entire batch gradient.
        self.ctc_criterion = nn.CTCLoss(
            blank=self.ctc_blank_id,
            reduction='mean',
            zero_infinity=True,
        )

        print(
            f"[CTCMixin] Initialised | hidden={hidden_size}, vocab={vocab_size}, "
            f"blank_id={self.ctc_blank_id}, weight={self.ctc_weight}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Loss computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_ctc_loss(
        self,
        visual_outputs: torch.Tensor,
        visual_masks:   torch.Tensor,
        samples:        Dict,
    ) -> torch.Tensor:
        """
        Compute the TxtCTC auxiliary loss.

        Call this from shared_step() during the main SLT training phase only
        (after warmup, split == 'train').  Returns a zero-gradient scalar if
        use_ctc is False so callers don't need an extra guard.

        Args:
            visual_outputs: [B, T, hidden_size] — post fusion_proj features,
                            the same tensor fed into the LLM.
            visual_masks:   [B, T] bool tensor — True = real (non-padding) token.
            samples:        Batch dict; must contain key 'text' (List[str]).

        Returns:
            Scalar float32 tensor: ctc_weight * L_TxtCTC.
            Callers can add this directly to the running loss without any
            further scaling.
        """
        if not self.use_ctc:
            return torch.tensor(0.0, device=visual_outputs.device)

        # ── 1. CTC log-probabilities ──────────────────────────────────────────
        # Cast to float32: log_softmax + CTC over ~32k vocab is numerically
        # unstable in bf16 (10-bit mantissa).
        logits    = self.ctc_head(visual_outputs.float())      # [B, T, vocab]
        log_probs = F.log_softmax(logits, dim=-1)              # [B, T, vocab]
        log_probs = log_probs.permute(1, 0, 2)                 # [T, B, vocab]  ← CTCLoss convention

        # ── 2. Tokenise targets ───────────────────────────────────────────────
        # padding=False → returns raw lists; we supply lengths separately.
        # truncation keeps targets consistent with max_txt_len used elsewhere.
        encoded = self.t5_tokenizer(
            samples['text'],
            padding=False,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors=None,
        )['input_ids']                                         # List[List[int]]

        # CTCLoss requires a flat 1-D target tensor (no padding allowed)
        targets_flat = torch.tensor(
            [tok for seq in encoded for tok in seq],
            dtype=torch.long,
            device=visual_outputs.device,
        )

        # ── 3. Sequence lengths ───────────────────────────────────────────────
        input_lengths  = visual_masks.sum(dim=1).long()        # [B] real visual tokens
        target_lengths = torch.tensor(
            [len(seq) for seq in encoded],
            dtype=torch.long,
            device=visual_outputs.device,
        )

        # ── 4. Diagnostic guard ───────────────────────────────────────────────
        # zero_infinity=True already handles impossible alignments gracefully,
        # but we surface a warning so it's visible in logs during debugging.
        n_invalid = (input_lengths < target_lengths).sum().item()
        if n_invalid > 0:
            print(
                f"[CTCMixin WARNING] {n_invalid} sample(s) have "
                f"input_length < target_length — their CTC contribution is "
                f"zeroed out by zero_infinity=True."
            )

        # ── 5. Loss ───────────────────────────────────────────────────────────
        ctc_loss = self.ctc_criterion(
            log_probs,       # [T, B, vocab]
            targets_flat,    # [sum(target_lengths)]
            input_lengths,   # [B]
            target_lengths,  # [B]
        )

        # Weight is baked in here so the caller just does: loss += ctc_loss
        return self.ctc_weight * ctc_loss