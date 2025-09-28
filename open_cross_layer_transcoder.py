from __future__ import annotations

import math, json, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, PreTrainedTokenizerFast
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# --------------------------------------------------------------------
# Utility functions and activation modules 
# --------------------------------------------------------------------

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class JumpReLU(nn.Module):
    def __init__(self, threshold: float, bandwidth: float) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)
    
    def extra_repr(self) -> str:
        return f"threshold={self.threshold.item():.3f}, bandwidth={self.bandwidth}"


class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, idx = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=idx, value=1)
        return x * gate.to(x.dtype)
    

# --------------------------------------------------------------------
#  Cross-Layer Transcoder
# --------------------------------------------------------------------


class OpenCrossLayerTranscoder(nn.Module):
    """Cross-layer transcoder (CLT) for GPT-2-style models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        num_features: int = 1536,
        device: str | torch.device = "cpu",
        activation_type: str = "jumprelu",
        topk_features: int | None = None,
        delta: float = 5e-4,
        c: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.delta, self.c = delta, c

        # backbone
        self.base_model = GPT2Model.from_pretrained(model_name).to(self.device)
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_layers = self.base_model.config.n_layer
        self.hidden_size = self.base_model.config.n_embd
        self.n_feat = num_features

        # encoders / decoders
        self.encoders = nn.ModuleList(
            [nn.Linear(self.hidden_size, num_features, bias=True) for _ in range(self.num_layers)]
        ).to(self.device)

        self.decoders = nn.ModuleDict()
        for enc_l in range(self.num_layers):
            for dec_l in range(enc_l, self.num_layers):
                self.decoders[f"{enc_l}_{dec_l}"] = nn.Linear(num_features, self.hidden_size, bias=True).to(
                    self.device
                )

        self._init_weights()

        # activation functions
        self.activation_functions = nn.ModuleList()
        for _ in range(self.num_layers):
            if activation_type == "jumprelu":
                self.activation_functions.append(JumpReLU(0.03, 1.0))
            elif activation_type == "topk":
                k = topk_features or max(1, num_features // 20)
                self.activation_functions.append(TopK(k))
            else:
                self.activation_functions.append(nn.ReLU())
        self.activation_functions = self.activation_functions.to(self.device)

        # hook storage
        self._hooks: List[Any] = []
        self._mlp_in: Dict[int, torch.Tensor] = {}
        self._mlp_out: Dict[int, torch.Tensor] = {}
        self._register_hooks()
        
    # ---------------------------------------------------------------
    # Weight initialisation (paper scheme – no sqrtL in decoder scale)
    # ---------------------------------------------------------------
    def _init_weights(self):
        enc_std = 1.0 / math.sqrt(self.n_feat)
        dec_std = 1.0 / math.sqrt(self.hidden_size)
        for enc in self.encoders:
            nn.init.uniform_(enc.weight, -enc_std, enc_std)
            nn.init.zeros_(enc.bias)
        for dec in self.decoders.values():
            nn.init.uniform_(dec.weight, -dec_std, dec_std)
            nn.init.zeros_(dec.bias)

    # ---------------------------------------------------------------
    # Hook registration
    # ---------------------------------------------------------------
    def _register_hooks(self):
        self._remove_hooks()

        def _make(idx):
            def hook(mod, inp, out):
                self._mlp_in[idx] = inp[0]
                self._mlp_out[idx] = out

            return hook
        
        # Handle both GPT2Model (has .h) and GPT2LMHeadModel (has .transformer.h)
        if hasattr(self.base_model, "h"):
            blocks = self.base_model.h
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            blocks = self.base_model.transformer.h
        else:
            raise AttributeError(
                "Could not find transformer blocks: expected `.h` or "
                "`.transformer.h` on the GPT‑2 backbone."
            )

        for i, block in enumerate(blocks):
            hook = block.mlp.register_forward_hook(_make(i))
            self._hooks.append(hook)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---------------------------------------------------------------
    # Encode and activate a single layer
    # ---------------------------------------------------------------
    def _encode_layer(self, layer: int, resid: torch.Tensor) -> torch.Tensor:
        z = self.encoders[layer](resid)
        return self.activation_functions[layer](z)

    # ---------------------------------------------------------------
    # Anthropic-style loss
    # ---------------------------------------------------------------
    def loss(self, residual_streams: Optional[List[torch.Tensor]] = None, *, debug: bool = False):
        """
        Computes reconstruction and sparsity loss against the cached MLP
        activations from the last forward pass.
        Returns
        -------
        (total_loss, mse_avg, l0_avg)
        """
        if residual_streams is not None:  # legacy path
            self._mlp_in = {i: residual_streams[i] for i in range(self.num_layers)}
            self._mlp_out = {i: residual_streams[i + 1] - residual_streams[i] for i in range(self.num_layers)}
        else:
            assert self._mlp_in, "Run a forward pass or supply residual_streams."

        MSE, Spar, L0 = 0.0, 0.0, 0.0
        feats_cache: Dict[int, torch.Tensor] = {}

        for l in range(self.num_layers):
            resid_pre = self._mlp_in[l]
            delta_mlp = self._mlp_out[l]

            f_l = self._encode_layer(l, resid_pre)
            feats_cache[l] = f_l

            # reconstruction from all previous encoders ≤ l
            Δ_hat = torch.zeros_like(delta_mlp)
            for j in range(l + 1):
                Δ_hat = Δ_hat + self.decoders[f"{j}_{l}"](feats_cache[j])

            mse = F.mse_loss(Δ_hat, delta_mlp)
            MSE += mse

            # sparsity
            dec_norm = torch.stack(
                [torch.norm(self.decoders[f"{l}_{k}"].weight, dim=0) for k in range(l, self.num_layers)]
            ).sum(0)
            sparsity = torch.tanh(self.c * dec_norm * f_l.abs()).mean()
            Spar = Spar + self.delta * sparsity

            L0 = L0 + (f_l.abs() > 1e-6).float().mean()

        total = MSE + Spar
        n = self.num_layers

        # --- debug: print overall sparsity once per call ---------------------
        # Use exact non-zero density as a sanity check (requested by user).
        with torch.no_grad():
            nz_elems, tot_elems = 0, 0
            for f in feats_cache.values():
                nz_elems += (f != 0).sum().item()
                tot_elems += f.numel()
            sparsity_ratio = nz_elems / tot_elems if tot_elems else 0.0
            print(f"[CLT.loss] non-zero activation ratio: {sparsity_ratio:.6f}")

        return total, MSE.item() / n, (L0 / n).item()

    # ---------------------------------------------------------------
    # Forward (features only – no replacement)
    # ---------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        self._mlp_in.clear()
        self._mlp_out.clear()
        outs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = {l: self._encode_layer(l, self._mlp_in[l]) for l in self._mlp_in}
        return outs, feats

    # ---------------------------------------------------------------
    # from_gpt2: reuse external backbone
    # ---------------------------------------------------------------
    @classmethod
    def from_gpt2(
        cls,
        gpt2_model: "GPT2Model",
        n_features_per_layer: int = 1536,
        delta: float = 5e-4,
        c: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> "OpenCrossLayerTranscoder":
        device = device or next(gpt2_model.parameters()).device
        inst = cls("gpt2", n_features_per_layer, device, delta=delta, c=c)
        inst.base_model = gpt2_model
        inst.num_layers = inst.base_model.config.n_layer
        inst.hidden_size = inst.base_model.config.n_embd

        # recreate encoders/decoders
        inst.encoders = nn.ModuleList(
            [nn.Linear(inst.hidden_size, n_features_per_layer, bias=True) for _ in range(inst.num_layers)]
        ).to(device)

        inst.decoders = nn.ModuleDict(
            {
                f"{i}_{j}": nn.Linear(n_features_per_layer, inst.hidden_size, bias=True).to(device)
                for i in range(inst.num_layers)
                for j in range(i, inst.num_layers)
            }
        )
        inst._init_weights()
        inst._register_hooks()
        inst.hparams = dict(n_features_per_layer=n_features_per_layer, delta=delta, c=c)
        return inst

    # ---------- checkpoint helpers ---------------------------------
    def save_model(self, path: str | Path):
        """
        Persist encoders, decoders and hyper-params in a single
        safetensors/pt file
        """
        torch.save(
            {
                "encoders": self.encoders.state_dict(),
                "decoders": self.decoders.state_dict(),
                "hparams": {"n_feat": self.n_feat, "delta": self.delta, "c": self.c},
            },
            str(path),
        )

    # ----- legacy convenience wrappers (Anthropic naming) ----------
    def save(self, path: str | Path):
        """Legacy alias used by training script."""
        return self.save_model(path)

    @classmethod
    def load_model(cls, path: str | Path, device: str | torch.device = "cpu", backbone: Optional[nn.Module] = None):
        if backbone is None:
            raise ValueError(
                "A `backbone` model must be provided to `load_model` to ensure "
                "the transcoder is initialized with the correct architecture."
            )

        chk = torch.load(str(path), map_location=device)
        
        # Create a dummy instance first
        # The `model_name` will be 'gpt2' by default, creating a temporary
        # backbone that we will immediately replace
        inst = cls(
            num_features=chk["hparams"]["n_feat"],
            device=device,
            delta=chk["hparams"]["delta"],
            c=chk["hparams"]["c"],
        )
        
        # Overwrite its state with the correct backbone and dimensions
        inst.base_model = backbone
        inst.num_layers = inst.base_model.config.n_layer
        inst.hidden_size = inst.base_model.config.n_embd
        n_feat = chk["hparams"]["n_feat"]
        
        # Re-create encoders/decoders with the correct dimensions
        inst.encoders = nn.ModuleList(
            [nn.Linear(inst.hidden_size, n_feat, bias=True) for _ in range(inst.num_layers)]
        ).to(device)

        inst.decoders = nn.ModuleDict(
            {
                f"{i}_{j}": nn.Linear(n_feat, inst.hidden_size, bias=True).to(device)
                for i in range(inst.num_layers)
                for j in range(i, inst.num_layers)
            }
        )
        
        # Now that the modules have the correct shapes, load state
        inst.encoders.load_state_dict(chk["encoders"])
        inst.decoders.load_state_dict(chk["decoders"])

        # Re-register the hooks on the correct model
        inst._register_hooks()

        return inst

    # keep the old name for backwards‑compat
    @classmethod
    def load(cls, path: str | Path, device="cpu", backbone: Optional[nn.Module] = None):
        return cls.load_model(path, device=device, backbone=backbone)


# --------------------------------------------------------------------
# Replacement model helper
# --------------------------------------------------------------------


class CLTReplacementModel(nn.Module):
    """
    Runs GPT-2 with a fixed CLT: each block's MLP output is replaced
    by the sum of decoder reconstructions from all previous encoders,
    exactly as in Anthropic's paper
    """

    def __init__(self, gpt2_lm, clt: OpenCrossLayerTranscoder, freeze_attn_ln: bool = True):
        super().__init__()
        self.lm = gpt2_lm
        self.clt = clt
        self.freeze_attn_ln = freeze_attn_ln
        self.feats_cache: Dict[int, torch.Tensor] = {}

        # hijack each block's MLP
        self._orig_forward = []
        for idx, block in enumerate(self.lm.transformer.h):

            self._orig_forward.append(block.mlp.forward)

            def make(j):
                def new_fwd(mlp_mod, x):
                    # x: post-LN, pre-MLP residual stream for layer j
                    self.clt._mlp_in[j] = x

                    recon = torch.zeros_like(x)
                    for k in range(j + 1):
                        f_k = self.feats_cache.get(k)
                        if f_k is None:
                            f_k = self.clt._encode_layer(k, self.clt._mlp_in[k])
                            self.feats_cache[k] = f_k
                        recon += self.clt.decoders[f"{k}_{j}"](f_k)
                    return recon

                return new_fwd

            block.mlp.forward = make(idx)

        if freeze_attn_ln:
            for blk in self.lm.transformer.h:
                blk.attn.register_forward_hook(lambda m, i, o: o.detach())
                blk.ln_1.register_forward_hook(lambda m, i, o: o.detach())

    def forward(self, *args, **kwargs):
        # Before running the main forward pass, clear caches
        # These will be populated during the pass by hooked MLPs
        self.clt._mlp_in.clear()
        self.feats_cache.clear()
        return self.lm(*args, **kwargs)


# convenience alias
CrossLayerTranscoder = OpenCrossLayerTranscoder