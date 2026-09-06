from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import random
from dataclasses import dataclass, field

import torch
from torch import nn

from .hif4 import (
    CompactHiF4Parameters,
    HIF4_BLOCK_SIZE,
    HiF4Parameters,
    expand_hif4_parameters,
    fit_hif4_compact_parameters,
    fit_hif4_parameters,
    project_hif4_source_to_target_cells,
    quantize_hif4_with_parameters,
)
from .config import resolve_quant_exempt_modules
from .hisq_rotation import (
    HiSQRotation,
    apply_hisq_rotation,
    derive_hisq_rotation,
    precondition_linear_for_hisq_rotation,
    resolve_hisq_block_size,
)
from .quantization import fake_quantize, replace_linear_with_fake_quant
from .smoothquant import smoothquant_input_scale


LOGGER = logging.getLogger(__name__)


@dataclass
class GPTQStats:
    samples: int
    mean_loss: float
    damp_multiplier: float
    # How often the HiF4 scale search picked each candidate step, keyed by step.
    # A search that never leaves 0 produces a checkpoint identical to the
    # reference one, and without this the only way to notice is to hash the
    # shards and guess why.
    scale_step_counts: dict[int, int] = field(default_factory=dict)


@torch.no_grad()
def _in_group_act_order_permutation(
    hessian: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Order columns by descending Hessian diagonal, within each group only.

    GPTQ gains from quantizing influential columns first, while many columns
    remain free to absorb the error. A global reordering is not available here:
    HiF4 derives its parameters from 64 positionally contiguous elements, and
    nothing at inference time records a reordering, so moving a column between
    groups changes what the deployed model computes. Permuting inside a group
    leaves group membership, and therefore the deployed arithmetic, untouched.
    """

    columns = hessian.shape[0]
    diagonal = hessian.diagonal().reshape(-1, group_size)
    offsets = torch.arange(
        0, columns, group_size, device=hessian.device
    ).unsqueeze(1)
    return (diagonal.argsort(dim=1, descending=True) + offsets).reshape(-1)


@torch.no_grad()
def _search_hif4_group_scale_steps(
    group_values: torch.Tensor,
    hinv_diagonal: torch.Tensor,
    steps: tuple[int, ...],
) -> torch.Tensor:
    """Pick each output row's HiF4 block scale by GPTQ's own error metric.

    Candidates are whole steps along the E6M2 grid, in both directions. HiF4
    rounds the scale to nearest rather than taking a ceiling, so about half of
    the blocks already clip at ``steps = 0`` and need a *larger* scale, while
    the rest have headroom and can trade clipping for a finer step -- which
    GPTQ is in a position to absorb, since the error it makes on a column is
    compensated on later columns.

    Scoring has to use GPTQ's own weighting, and get its power right. ``hinv``
    is the upper Cholesky factor U of the inverse Hessian, and the solver
    propagates ``(w - q) / U_jj``, so a column's contribution to the objective
    is ``dw^2 / U_jj^2``. Weighting by ``1/U_jj`` instead ranks candidates by a
    different objective than the one being minimised -- it still prefers
    cheaper columns, so a test that only checks the direction cannot tell the
    two apart, but it prices them wrongly relative to each other.

    The choice is per output row, since each row has its own block maximum.
    """

    rows = group_values.shape[0]
    weights = (
        hinv_diagonal.to(torch.float32).pow(-2).clamp_min(0.0).to(group_values.dtype)
    )
    best_score = torch.full(
        (rows,), float("inf"), device=group_values.device, dtype=group_values.dtype
    )
    best_steps = torch.zeros(
        (rows, 1, 1, 1), device=group_values.device, dtype=torch.float32
    )
    for step in steps:
        candidate = fit_hif4_compact_parameters(group_values, scale_steps=float(step))
        expanded = expand_hif4_parameters(candidate, block_shape=group_values.shape)
        quantized = quantize_hif4_with_parameters(group_values, expanded)
        score = (((group_values - quantized) ** 2) * weights).sum(dim=1)
        improved = score < best_score
        best_score = torch.where(improved, score, best_score)
        best_steps = torch.where(
            improved.view(-1, 1, 1, 1),
            torch.full_like(best_steps, float(step)),
            best_steps,
        )
    return best_steps


class GPTQ:
    """Layer-local GPTQ using the reference blockwise error propagation."""

    def __init__(
        self, layer: nn.Linear, *, hessian_block_size: int | None = None
    ) -> None:
        if not isinstance(layer, nn.Linear):
            raise TypeError("GPTQ expects nn.Linear")
        self.layer = layer
        self.columns = layer.in_features
        self.hessian_block_size = hessian_block_size
        if hessian_block_size is None:
            hessian_shape = (self.columns, self.columns)
        else:
            if hessian_block_size <= 0 or self.columns % hessian_block_size:
                raise ValueError(
                    "hessian_block_size must be a positive divisor of in_features"
                )
            hessian_shape = (
                self.columns // hessian_block_size,
                hessian_block_size,
                hessian_block_size,
            )
        self.hessian = torch.zeros(
            hessian_shape, device=layer.weight.device, dtype=torch.float32
        )
        self.samples = 0
        self.weight_override: torch.Tensor | None = None

    def set_weight_override(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != tuple(self.layer.weight.shape):
            raise ValueError("GPTQ weight override shape mismatch")
        if not torch.isfinite(weight).all():
            raise ValueError("GPTQ weight override must be finite")
        self.weight_override = weight.detach().float()

    def _source_weight(self) -> torch.Tensor:
        if self.weight_override is not None:
            return self.weight_override.clone()
        return self.layer.weight.detach().float().clone()

    @torch.inference_mode()
    def add_batch(self, inputs: torch.Tensor) -> None:
        inputs = inputs.reshape(-1, inputs.shape[-1]).float()
        if self.hessian_block_size is None:
            self.hessian.add_(inputs.T @ inputs)
        else:
            grouped = inputs.reshape(inputs.shape[0], -1, self.hessian_block_size)
            self.hessian.add_(torch.einsum("ngi,ngj->gij", grouped, grouped))
        self.samples += inputs.shape[0]

    @staticmethod
    def _damped_inverse_cholesky(
        hessian: torch.Tensor,
        base_damp: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Return GPTQ's upper inverse factor with end-to-end damping retries."""

        diagonal = torch.arange(hessian.shape[-1], device=hessian.device)
        for multiplier in (1.0, 10.0, 100.0, 1000.0):
            candidate = hessian.clone()
            candidate[..., diagonal, diagonal] += base_damp[..., None] * multiplier
            factor, info = torch.linalg.cholesky_ex(candidate)
            if torch.any(info):
                continue
            inverse = torch.cholesky_inverse(factor)
            inverse = (inverse + inverse.transpose(-1, -2)) * 0.5
            inverse_factor, inverse_info = torch.linalg.cholesky_ex(inverse, upper=True)
            if not torch.any(inverse_info) and torch.isfinite(inverse_factor).all():
                return inverse_factor, multiplier
        raise RuntimeError(
            "GPTQ Hessian or its inverse is not positive definite after damping"
        )

    @torch.inference_mode()
    def _quantize_block_diagonal(
        self,
        *,
        quant_format: str,
        bits: int,
        group_size: int,
        symmetric: bool,
        damp: float,
        capture_hif4_metadata: bool,
        capture_latent_master: bool,
    ) -> GPTQStats:
        """Batched group-local GPTQ for very wide projection matrices."""

        if self.hessian_block_size != group_size:
            raise ValueError(
                "block-diagonal GPTQ requires Hessian blocks to match weight groups"
            )
        weight = self._source_weight()
        original = weight.clone()
        groups = self.columns // group_size
        hessian = self.hessian / self.samples
        dead = torch.diagonal(hessian, dim1=-2, dim2=-1) == 0
        diagonal = torch.arange(group_size, device=weight.device)
        hessian[:, diagonal, diagonal] = torch.where(
            dead,
            torch.ones_like(hessian[:, diagonal, diagonal]),
            hessian[:, diagonal, diagonal],
        )

        work = weight.reshape(weight.shape[0], groups, group_size)
        work.masked_fill_(dead.unsqueeze(0), 0)
        compact_hif4_parameters = None
        hif4_parameters = None
        if quant_format == "hif4":
            if capture_hif4_metadata:
                compact_hif4_parameters = fit_hif4_compact_parameters(work)
                hif4_parameters = expand_hif4_parameters(
                    compact_hif4_parameters,
                    block_shape=work.shape,
                )
            else:
                hif4_parameters = fit_hif4_parameters(work)
        base_damp = damp * torch.diagonal(hessian, dim1=-2, dim2=-1).mean(
            dim=-1
        ).clamp_min(1e-8)
        hinv, damp_multiplier = self._damped_inverse_cholesky(hessian, base_damp)
        quantized = torch.zeros_like(work)
        # All independent weight groups share an offset and are updated in one
        # GPU operation, avoiding one Python loop per input column.
        for offset in range(group_size):
            flat_groups = work.reshape(-1, group_size)
            column = work[:, :, offset].reshape(-1)
            if hif4_parameters is None:
                qcolumn = self._quantize_column(
                    column,
                    flat_groups,
                    bits=bits,
                    symmetric=symmetric,
                )
            else:
                column_parameters = HiF4Parameters(
                    quant_multiplier=(
                        hif4_parameters.quant_multiplier[:, :, offset].reshape(-1)
                    ),
                    dequant_scale=(
                        hif4_parameters.dequant_scale[:, :, offset].reshape(-1)
                    ),
                )
                qcolumn = quantize_hif4_with_parameters(column, column_parameters)
            qcolumn = qcolumn.reshape(weight.shape[0], groups)
            quantized[:, :, offset] = qcolumn
            error = (work[:, :, offset] - qcolumn) / hinv[:, offset, offset].unsqueeze(
                0
            )
            work[:, :, offset:] -= error.unsqueeze(-1) * hinv[
                :, offset, offset:
            ].unsqueeze(0)

        quantized_weight = quantized.reshape_as(weight)
        self.layer.weight.data.copy_(quantized_weight.to(self.layer.weight.dtype))
        if capture_hif4_metadata:
            if compact_hif4_parameters is None:
                raise RuntimeError("cannot capture metadata for a non-HiF4 quantizer")
            self.layer.faquant_hif4_qat_parameters = compact_hif4_parameters
            if capture_latent_master:
                latent = project_hif4_source_to_target_cells(
                    original.reshape_as(work),
                    quantized,
                    compact_hif4_parameters,
                    output_dtype=self.layer.weight.dtype,
                ).reshape_as(original)
                self.layer.faquant_hif4_qat_master = latent.to(
                    device="cpu", dtype=self.layer.weight.dtype
                )
        weighted_error = (original - quantized_weight).square().mean().item()
        self.hessian = torch.empty(0, device=self.layer.weight.device)
        return GPTQStats(
            samples=self.samples,
            mean_loss=weighted_error,
            damp_multiplier=damp_multiplier,
        )

    @staticmethod
    def _quantize_column(
        column: torch.Tensor,
        group: torch.Tensor,
        *,
        bits: int,
        symmetric: bool,
    ) -> torch.Tensor:
        if symmetric:
            qmax = 2 ** (bits - 1) - 1
            qmin = -(2 ** (bits - 1))
            scale = group.abs().amax(dim=1).clamp_min(1e-8) / qmax
            return torch.round(column / scale).clamp_(qmin, qmax) * scale
        qmax = 2**bits - 1
        xmin = torch.minimum(
            group.amin(dim=1), torch.zeros(group.shape[0], device=group.device)
        )
        xmax = torch.maximum(
            group.amax(dim=1), torch.zeros(group.shape[0], device=group.device)
        )
        scale = ((xmax - xmin) / qmax).clamp_min(1e-8)
        zero = torch.round(-xmin / scale).clamp_(0, qmax)
        return (torch.round(column / scale + zero).clamp_(0, qmax) - zero) * scale

    @torch.inference_mode()
    def quantize(
        self,
        *,
        quant_format: str = "int",
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        damp: float = 0.01,
        block_size: int = 128,
        capture_hif4_metadata: bool = False,
        capture_latent_master: bool = False,
        scale_search_steps: tuple[int, ...] | None = None,
        act_order_within_group: bool = False,
    ) -> GPTQStats:
        if self.samples == 0:
            raise RuntimeError("GPTQ requires calibration samples")
        if block_size <= 0:
            raise ValueError("GPTQ block_size must be positive")
        if quant_format not in ("int", "hif4"):
            raise ValueError(f"unsupported quant_format={quant_format!r}")
        if group_size == -1:
            group_size = self.columns
        if quant_format == "hif4":
            if bits != 4 or not symmetric:
                raise ValueError("HiF4 GPTQ requires 4-bit sign-magnitude quantization")
            if group_size != HIF4_BLOCK_SIZE:
                raise ValueError("HiF4 GPTQ requires group_size=64")
            if block_size % HIF4_BLOCK_SIZE:
                raise ValueError("HiF4 GPTQ block_size must be a multiple of 64")
        elif capture_hif4_metadata:
            raise ValueError("HiF4 metadata capture requires quant_format='hif4'")
        if capture_latent_master and not capture_hif4_metadata:
            raise ValueError("latent-master capture requires HiF4 metadata")
        if self.columns % group_size:
            raise ValueError("GPTQ group_size must divide layer.in_features")
        if scale_search_steps is not None and quant_format != "hif4":
            raise ValueError("scale search is only defined for HiF4")
        if act_order_within_group and quant_format != "hif4":
            raise ValueError("in-group act-order is only wired for HiF4 groups")
        if self.hessian_block_size is not None:
            if act_order_within_group:
                raise ValueError(
                    "in-group act-order is not implemented on the "
                    "block-diagonal path"
                )
            if scale_search_steps is not None:
                raise ValueError(
                    "scale search is not implemented on the block-diagonal path"
                )
            return self._quantize_block_diagonal(
                quant_format=quant_format,
                bits=bits,
                group_size=group_size,
                symmetric=symmetric,
                damp=damp,
                capture_hif4_metadata=capture_hif4_metadata,
                capture_latent_master=capture_latent_master,
            )

        weight = self._source_weight()
        original = weight.clone()
        hessian = self.hessian / self.samples
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0
        act_perm = None
        act_inverse = None
        if act_order_within_group:
            act_perm = _in_group_act_order_permutation(hessian, group_size)
            act_inverse = torch.empty_like(act_perm)
            act_inverse[act_perm] = torch.arange(
                self.columns, device=act_perm.device
            )
            hessian = hessian[act_perm][:, act_perm]
            weight = weight[:, act_perm]
        base_damp = damp * torch.diag(hessian).mean().clamp_min(1e-8)
        # Numerical rank deficiency is common with short calibration runs. The
        # inverse can remain too ill-conditioned for its upper Cholesky even
        # when the first factorization succeeds, so retry the complete chain.
        hinv, damp_multiplier = self._damped_inverse_cholesky(hessian, base_damp)
        quantized = torch.zeros_like(weight)
        scale_step_counts: dict[int, int] = {}
        compact_scale = None
        compact_reciprocal = None
        compact_lv2_exponent = None
        compact_lv3_exponent = None
        if capture_hif4_metadata:
            groups = self.columns // HIF4_BLOCK_SIZE
            leading = (weight.shape[0], groups)
            compact_scale = torch.empty(
                (*leading, 1, 1, 1), device=weight.device, dtype=torch.float32
            )
            compact_reciprocal = torch.empty_like(compact_scale)
            compact_lv2_exponent = torch.empty(
                (*leading, 8, 1, 1), device=weight.device, dtype=torch.int8
            )
            compact_lv3_exponent = torch.empty(
                (*leading, 8, 2, 1), device=weight.device, dtype=torch.int8
            )

        for block_start in range(0, self.columns, block_size):
            block_stop = min(block_start + block_size, self.columns)
            block = weight[:, block_start:block_stop].clone()
            block_quant = torch.zeros_like(block)
            errors = torch.zeros_like(block)
            local_hinv = hinv[block_start:block_stop, block_start:block_stop]
            hif4_parameters: HiF4Parameters | None = None

            for offset in range(block_stop - block_start):
                column_index = block_start + offset
                group_start = (column_index // group_size) * group_size
                group_stop = group_start + group_size
                column = block[:, offset]
                if quant_format == "hif4":
                    offset_in_group = column_index - group_start
                    if offset_in_group == 0:
                        local_group_start = group_start - block_start
                        group_values = block[
                            :,
                            local_group_start : local_group_start + HIF4_BLOCK_SIZE,
                        ]
                        group_hinv_diagonal = local_hinv.diagonal()[
                            local_group_start
                            : local_group_start + HIF4_BLOCK_SIZE
                        ]
                        # HiF4's two micro-exponent levels are positional: the
                        # 64 values reshape to (8, 2, 4), so which elements share
                        # an exponent depends on where they sit. The deployed
                        # model reads the weight in storage order, so the fit has
                        # to see storage order even while the solver visits
                        # columns by descending Hessian. Both the values and
                        # their error weights move together.
                        visit_to_storage = None
                        if act_perm is not None:
                            visit_to_storage = (
                                act_perm[group_start:group_stop] - group_start
                            )
                            storage_to_visit = torch.argsort(visit_to_storage)
                            group_values = group_values[:, storage_to_visit]
                            group_hinv_diagonal = group_hinv_diagonal[
                                storage_to_visit
                            ]
                        group_scale_steps = 0.0
                        if scale_search_steps is not None:
                            group_scale_steps = _search_hif4_group_scale_steps(
                                group_values,
                                group_hinv_diagonal,
                                scale_search_steps,
                            )
                            for step in scale_search_steps:
                                chosen = int(
                                    (group_scale_steps == float(step)).sum()
                                )
                                if chosen:
                                    scale_step_counts[step] = (
                                        scale_step_counts.get(step, 0) + chosen
                                    )
                        if capture_hif4_metadata:
                            compact_group = fit_hif4_compact_parameters(
                                group_values, scale_steps=group_scale_steps
                            )
                            hif4_parameters = expand_hif4_parameters(
                                compact_group,
                                block_shape=group_values.shape,
                            )
                            group_index = column_index // HIF4_BLOCK_SIZE
                            compact_scale[:, group_index] = compact_group.scale
                            compact_reciprocal[:, group_index] = (
                                compact_group.reciprocal
                            )
                            compact_lv2_exponent[:, group_index] = (
                                compact_group.scale_lv2_exponent
                            )
                            compact_lv3_exponent[:, group_index] = (
                                compact_group.scale_lv3_exponent
                            )
                        elif scale_search_steps is None:
                            hif4_parameters = fit_hif4_parameters(group_values)
                        else:
                            # Only the search needs the compact route, which is
                            # the one that can take a stepped block scale.
                            hif4_parameters = expand_hif4_parameters(
                                fit_hif4_compact_parameters(
                                    group_values, scale_steps=group_scale_steps
                                ),
                                block_shape=group_values.shape,
                            )
                        if visit_to_storage is not None:
                            # The expanded parameters are per storage position;
                            # the loop below indexes them by visit position.
                            hif4_parameters = HiF4Parameters(
                                quant_multiplier=(
                                    hif4_parameters.quant_multiplier[
                                        :, visit_to_storage
                                    ]
                                ),
                                dequant_scale=(
                                    hif4_parameters.dequant_scale[
                                        :, visit_to_storage
                                    ]
                                ),
                            )
                    if hif4_parameters is None:
                        raise RuntimeError("HiF4 GPTQ metadata was not initialized")
                    qcolumn = quantize_hif4_with_parameters(
                        column,
                        HiF4Parameters(
                            quant_multiplier=(
                                hif4_parameters.quant_multiplier[:, offset_in_group]
                            ),
                            dequant_scale=(
                                hif4_parameters.dequant_scale[:, offset_in_group]
                            ),
                        ),
                    )
                else:
                    qcolumn = self._quantize_column(
                        column,
                        weight[:, group_start:group_stop],
                        bits=bits,
                        symmetric=symmetric,
                    )
                block_quant[:, offset] = qcolumn
                divisor = local_hinv[offset, offset]
                error = (column - qcolumn) / divisor
                block[:, offset:] -= error.unsqueeze(1) @ local_hinv[
                    offset, offset:
                ].unsqueeze(0)
                errors[:, offset] = error

            quantized[:, block_start:block_stop] = block_quant
            weight[:, block_stop:] -= errors @ hinv[block_start:block_stop, block_stop:]

        if act_inverse is not None:
            # Back to storage order. Everything downstream -- the saved
            # weight, the latent master, the parity check -- expects it.
            quantized = quantized[:, act_inverse]
        self.layer.weight.data.copy_(quantized.to(self.layer.weight.dtype))
        if capture_hif4_metadata:
            compact_parameters = CompactHiF4Parameters(
                scale=compact_scale,
                reciprocal=compact_reciprocal,
                scale_lv2_exponent=compact_lv2_exponent,
                scale_lv3_exponent=compact_lv3_exponent,
            )
            self.layer.faquant_hif4_qat_parameters = compact_parameters
            if capture_latent_master:
                grouped_shape = (
                    original.shape[0],
                    original.shape[1] // HIF4_BLOCK_SIZE,
                    HIF4_BLOCK_SIZE,
                )
                latent = project_hif4_source_to_target_cells(
                    original.reshape(grouped_shape),
                    quantized.reshape(grouped_shape),
                    compact_parameters,
                    output_dtype=self.layer.weight.dtype,
                ).reshape_as(original)
                self.layer.faquant_hif4_qat_master = latent.to(
                    device="cpu", dtype=self.layer.weight.dtype
                )
        weighted_error = (original - quantized).square().mean().item()
        self.hessian = torch.empty(0, device=self.layer.weight.device)
        return GPTQStats(
            samples=self.samples,
            mean_loss=weighted_error,
            damp_multiplier=damp_multiplier,
            scale_step_counts=scale_step_counts,
        )


CALIBRATION_CORPORA = ("wikitext2", "c4", "redpajama")

# Enough documents that the sampled windows do not repeat material, without
# tokenizing a whole shard. WikiText-2 train is ~2.4M tokens, so this is the
# same order of magnitude and keeps the corpora comparable in size.
_STREAMED_CALIBRATION_DOCUMENTS = 20000

REDPAJAMA_CALIBRATION_JSONL = "data/qad_redpajama/train.jsonl"


def _calibration_corpus_text(corpus: str, *, seed: int) -> str:
    """Return one plain-text blob per corpus, joined the same way for each.

    The point of swapping corpora is to vary the calibration distribution and
    nothing else, so every corpus goes through the identical
    join/tokenize/window path. In particular RedPajama is read as raw document
    text rather than through the chat template the QAD loader applies -- the
    template would change the token distribution as much as the corpus does and
    confound the comparison.
    """

    if corpus == "wikitext2":
        from datasets import load_dataset

        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
        )
        return "\n\n".join(dataset["text"])

    if corpus == "c4":
        from datasets import load_dataset

        stream = load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        ).shuffle(seed=seed, buffer_size=10000)
        documents = [
            record["text"]
            for _, record in zip(range(_STREAMED_CALIBRATION_DOCUMENTS), stream)
        ]
        return "\n\n".join(documents)

    if corpus == "redpajama":
        source = Path(REDPAJAMA_CALIBRATION_JSONL)
        if not source.exists():
            raise FileNotFoundError(
                f"RedPajama calibration corpus not found at {source}"
            )
        documents: list[str] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if len(documents) >= _STREAMED_CALIBRATION_DOCUMENTS:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                # The QAD build split each document into a user/assistant pair;
                # rejoining the turns recovers the original text.
                documents.append(
                    "".join(
                        str(message.get("content", ""))
                        for message in record.get("messages", [])
                    )
                )
        if not documents:
            raise ValueError(f"{source} contains no calibration records")
        return "\n\n".join(documents)

    raise ValueError(
        f"unsupported calibration corpus {corpus!r}, expected one of "
        f"{CALIBRATION_CORPORA}"
    )


CALIBRATION_CACHE_DIR = Path("data/cache/gptq_calibration")


def _encoded_calibration_corpus(
    tokenizer: object, corpus: str, *, seed: int
) -> torch.Tensor:
    """Tokenize a corpus once and reuse it.

    Streaming and tokenizing C4 costs minutes, and a sweep runs several builds
    of the same corpus concurrently. The cache keys on the seed because C4's
    document selection depends on it.
    """

    cache = CALIBRATION_CACHE_DIR / f"{corpus}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    encoded = tokenizer(
        _calibration_corpus_text(corpus, seed=seed),
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    ).input_ids[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a build that starts mid-write cannot read a
    # truncated tensor and calibrate on a silently shortened corpus.
    staging = cache.with_suffix(f".{os.getpid()}.tmp")
    torch.save(encoded, staging)
    staging.replace(cache)
    return encoded


def calibration_tokens(
    tokenizer: object,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    corpus: str = "wikitext2",
) -> torch.Tensor:
    """Build deterministic token windows for GPTQ calibration."""

    encoded = _encoded_calibration_corpus(tokenizer, corpus, seed=seed)
    if encoded.numel() <= seqlen:
        raise RuntimeError("calibration corpus is shorter than gptq_seqlen")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(
        0, encoded.numel() - seqlen, (nsamples,), generator=generator
    )
    return torch.stack([encoded[start : start + seqlen] for start in starts.tolist()])


def calibration_tokens_from_jsonl(
    tokenizer: object,
    path: str | Path,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    max_record_tokens: int = 512,
) -> torch.Tensor:
    """Pack deterministic, category-balanced chat records for GPTQ.

    This is intended for generic/reasoning calibration corpora rather than
    benchmark examples.  Records are balanced by their explicit ``category``
    field, shuffled with a local RNG, chat-templated including the assistant
    answer, and capped so that one long derivation cannot dominate the Hessian.
    Independent records are separated by EOS before being packed into fixed
    windows.
    """

    if nsamples <= 0 or seqlen <= 0 or max_record_tokens <= 0:
        raise ValueError("GPTQ JSONL calibration lengths must be positive")
    source = Path(path)
    categories: dict[str, list[list[dict[str, str]]]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"{source}:{line_number} must contain non-empty messages"
                )
            category = str(record.get("category", "uncategorized"))
            categories.setdefault(category, []).append(messages)
    if not categories:
        raise ValueError(f"{source} contains no calibration records")

    rng = random.Random(seed)
    names = sorted(categories)
    for records in categories.values():
        rng.shuffle(records)
    offsets = {name: 0 for name in names}
    eos_token_id = int(tokenizer.eos_token_id)
    target_tokens = nsamples * seqlen
    packed: list[int] = []
    category_index = 0
    while len(packed) < target_tokens:
        name = names[category_index % len(names)]
        category_index += 1
        records = categories[name]
        offset = offsets[name]
        if offset == len(records):
            rng.shuffle(records)
            offset = 0
        messages = records[offset]
        offsets[name] = offset + 1
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not token_ids:
            continue
        packed.extend(int(token) for token in token_ids[:max_record_tokens])
        packed.append(eos_token_id)
    return torch.tensor(packed[:target_tokens], dtype=torch.long).reshape(
        nsamples, seqlen
    )


class _CaptureComplete(Exception):
    pass


@torch.inference_mode()
def gptq_quantize_qwen3(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: object,
) -> dict[str, GPTQStats]:
    """Run layerwise GPTQ while keeping Qwen3's configured attention backend."""

    layers = model.model.layers
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    nsamples, seqlen = input_ids.shape
    hidden_size = model.config.hidden_size
    captured = torch.empty((nsamples, seqlen, hidden_size), device=device, dtype=dtype)
    outputs = torch.empty_like(captured)
    stats: dict[str, GPTQStats] = {}

    def capture_first_layer_inputs() -> dict[str, object]:
        cache: dict[str, object] = {"index": 0, "kwargs": None}
        original_first = layers[0]

        class Catcher(nn.Module):
            def __init__(self, module: nn.Module) -> None:
                super().__init__()
                self.module = module
                self.attention_type = module.attention_type

            def forward(
                self, hidden_states: torch.Tensor, **kwargs: object
            ) -> torch.Tensor:
                index = int(cache["index"])
                captured[index].copy_(hidden_states[0])
                cache["index"] = index + 1
                cache["kwargs"] = kwargs
                raise _CaptureComplete

        layers[0] = Catcher(original_first)
        try:
            for row in input_ids:
                try:
                    model(row.unsqueeze(0).to(device), use_cache=False)
                except _CaptureComplete:
                    pass
        finally:
            layers[0] = original_first
        if int(cache["index"]) != nsamples:
            raise RuntimeError(
                f"captured {cache['index']} Qwen samples, expected {nsamples}"
            )
        kwargs = dict(cache["kwargs"] or {})
        kwargs["use_cache"] = False
        return kwargs

    layer_kwargs = capture_first_layer_inputs()

    if config.quant_target == "attention":
        groups = (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ("self_attn.o_proj",),
        )
    elif config.quant_target == "ffn":
        groups = (("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
    elif config.quant_target == "all":
        groups = (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ("self_attn.o_proj",),
            ("mlp.gate_proj", "mlp.up_proj"),
            ("mlp.down_proj",),
        )
    else:
        return stats

    def resolve_module(root: nn.Module, path: str) -> nn.Module:
        """Resolve a projection without insisting it is still unquantized.

        Sequential collection quantizes a group before the next group's Hessian
        is built, so by then an earlier projection is a FakeQuantLinear.
        Callers that only need to know what is there use this; ``resolve``
        keeps the assertion for callers that require a Linear.
        """

        module: nn.Module = root
        for part in path.split("."):
            module = getattr(module, part)
        if getattr(module, "faquant_online_hadamard", False):
            module = module.module
        return module

    def resolve(root: nn.Module, path: str) -> nn.Linear:
        module = resolve_module(root, path)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"GPTQ expected nn.Linear at {path}")
        return module

    def install_fake_quant(root: nn.Module, path: str) -> None:
        parent = root
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replace_linear_with_fake_quant(
            parent, parts[-1], config, weights_prequantized=True
        )

    smoothquant_enabled = bool(getattr(config, "qwen_smoothquant", False))
    smoothquant_alpha = float(getattr(config, "qwen_smoothquant_alpha", 0.9))
    hisq_rotation_enabled = bool(
        getattr(config, "qwen_hisq_input_rotation", False)
    )
    hisq_rotation_block_size = int(
        getattr(config, "qwen_hisq_rotation_block_size", 1024)
    )
    hisq_rotation_down_proj_block_size = getattr(
        config, "qwen_hisq_rotation_block_size_down_proj", None
    )
    hisq_rotation_seed = int(getattr(config, "qwen_hisq_rotation_seed", 17))
    smoothquant_absmax: dict[str, torch.Tensor] = {}
    smoothquant_report: dict[str, object] = {}
    hisq_rotation_report: dict[str, object] = {
        "source": "GCC-HiFloat/HiSQRot4",
        "source_commit": "84cdcf7b393d2665eedb5cf26e11f590f6852c46",
        "construction": "per-linear sign -> permutation -> block-Hadamard",
        "block_size": hisq_rotation_block_size,
        "down_proj_block_size": hisq_rotation_down_proj_block_size,
        "seed": hisq_rotation_seed,
        "layers": {},
    }

    if smoothquant_enabled:
        # Match HiSQRot4 Stage 1: collect raw BF16 per-channel input ranges
        # before SmoothQuant or GPTQ changes the model. Explicit attention
        # output QDQ is disabled so o_proj also observes its raw BF16 input.
        for layer_index, layer in enumerate(layers):
            handles = []
            for group in groups:
                for name in group:
                    key = f"model.layers.{layer_index}.{name}"

                    def collect_absmax(
                        _module: nn.Module,
                        args: tuple[torch.Tensor, ...],
                        _out: torch.Tensor,
                        stat_key: str = key,
                    ) -> None:
                        activations = args[0].detach()
                        dims = tuple(range(activations.ndim - 1))
                        current = activations.abs().amax(dim=dims).float()
                        previous = smoothquant_absmax.get(stat_key)
                        smoothquant_absmax[stat_key] = (
                            current
                            if previous is None
                            else torch.maximum(previous, current)
                        )

                    handles.append(
                        resolve(layer, name).register_forward_hook(collect_absmax)
                    )

            attention = layer.self_attn
            original_output_quant = bool(
                getattr(attention, "faquant_attention_output_quant", False)
            )
            attention.faquant_attention_output_quant = False
            try:
                for sample_index, sample in enumerate(captured):
                    outputs[sample_index].copy_(
                        layer(sample.unsqueeze(0), **layer_kwargs)[0]
                    )
            finally:
                attention.faquant_attention_output_quant = original_output_quant
                for handle in handles:
                    handle.remove()
            captured, outputs = outputs, captured
            LOGGER.info(
                "Qwen SmoothQuant collected layer %d/%d",
                layer_index + 1,
                len(layers),
            )

        expected_stats = len(layers) * sum(len(group) for group in groups)
        if len(smoothquant_absmax) != expected_stats:
            raise RuntimeError(
                "Qwen SmoothQuant activation coverage mismatch: "
                f"got {len(smoothquant_absmax)}, expected {expected_stats}"
            )
        # The propagation pass reuses both work buffers. Re-capture the
        # original first-layer inputs before starting quantized propagation.
        layer_kwargs = capture_first_layer_inputs()
        smoothquant_report = {
            "source": "GCC-HiFloat/HiSQRot4",
            "alpha": smoothquant_alpha,
            "eps": 1e-5,
            "calibration_samples": int(nsamples),
            "calibration_seqlen": int(seqlen),
            "layers": {},
        }

    act_order_within_group = bool(
        getattr(config, "gptq_act_order_within_group", False)
    )
    raw_steps = getattr(config, "gptq_hif4_scale_search_steps", ()) or ()
    scale_search_steps = tuple(int(s) for s in raw_steps) or None
    damp_overrides = {
        str(name): float(value)
        for name, value in (
            getattr(config, "gptq_damp_overrides", ()) or ()
        )
    }
    mxfp8_hessian_fraction = float(
        getattr(config, "gptq_mxfp8_hessian_fraction", 1.0)
    )
    mxfp8_hessian_mode = str(
        getattr(config, "gptq_mxfp8_hessian_mode", "paired")
    )
    sequential_groups = bool(getattr(config, "gptq_sequential_groups", False))
    if sequential_groups and not getattr(
        config, "gptq_propagate_fake_activations", True
    ):
        # Without fake-quant propagation a later group's forward still uses the
        # unquantized earlier weights, so the extra passes would cost time and
        # change nothing.
        raise ValueError(
            "gptq_sequential_groups requires gptq_propagate_fake_activations"
        )
    exempt_modules = resolve_quant_exempt_modules(config, len(layers))
    for layer_index, layer in enumerate(layers):
        # Drop this layer's exempt projections from the work list. Everything
        # below iterates the filtered groups, so an exempt projection gets no
        # rotation, no Hessian, no quantization and no FakeQuantLinear.
        layer_groups = tuple(
            filtered
            for filtered in (
                tuple(
                    name
                    for name in group
                    if (layer_index, name.rsplit(".", 1)[-1]) not in exempt_modules
                )
                for group in groups
            )
            if filtered
        )
        if not layer_groups:
            # Still propagate. Every later layer's Hessian must be built from
            # what this layer actually emits, and a fully exempt layer emits
            # its BF16 output, so skipping the forward pass here would
            # calibrate the rest of the stack against a layer that never runs.
            for sample_index, sample in enumerate(captured):
                outputs[sample_index].copy_(
                    layer(sample.unsqueeze(0), **layer_kwargs)[0]
                )
            captured, outputs = outputs, captured
            LOGGER.info(
                "Qwen GPTQ left layer %d/%d in BF16 (exempt)",
                layer_index + 1,
                len(layers),
            )
            continue
        layer_rotations: dict[str, HiSQRotation] = {}
        rotated_weight_overrides: dict[str, torch.Tensor] = {}
        if hisq_rotation_enabled:
            for group in layer_groups:
                for name in group:
                    key = f"model.layers.{layer_index}.{name}"
                    linear = resolve(layer, name)
                    rotation = derive_hisq_rotation(
                        key,
                        linear.in_features,
                        seed=hisq_rotation_seed,
                        block_size=resolve_hisq_block_size(
                            key,
                            linear.in_features,
                            block_size=hisq_rotation_block_size,
                            down_proj_block_size=(
                                hisq_rotation_down_proj_block_size
                            ),
                        ),
                        device=linear.weight.device,
                    )
                    layer_rotations[name] = rotation
                    rotated_weight_overrides[name] = (
                        precondition_linear_for_hisq_rotation(linear, rotation)
                    )
                    hisq_rotation_report["layers"][key] = {
                        "input_features": linear.in_features,
                        "block_size": rotation.block_size,
                        "blocks": linear.in_features // rotation.block_size,
                        "layer_seed": rotation.layer_seed,
                    }

        quantizers = {
            name: GPTQ(resolve(layer, name)) for group in layer_groups for name in group
        }
        layer_scales: dict[str, torch.Tensor] = {}
        if smoothquant_enabled:
            for group in layer_groups:
                for name in group:
                    key = f"model.layers.{layer_index}.{name}"
                    scale = smoothquant_input_scale(
                        resolve(layer, name).weight,
                        smoothquant_absmax[key],
                        alpha=smoothquant_alpha,
                    )
                    layer_scales[name] = scale
                    smoothquant_report["layers"][key] = {
                        "input_features": int(scale.numel()),
                        "scale_min": float(scale.min().item()),
                        "scale_max": float(scale.max().item()),
                        "activation_absmax_min": float(
                            smoothquant_absmax[key].min().item()
                        ),
                        "activation_absmax_max": float(
                            smoothquant_absmax[key].max().item()
                        ),
                    }

        def install_rotation_hooks() -> list[object]:
            installed: list[object] = []
            if not hisq_rotation_enabled:
                return installed
            for name, rotation in layer_rotations.items():
                # Under sequential collection an earlier group is already a
                # FakeQuantLinear, which carries the rotation itself -- it
                # reads the metadata precondition_linear_for_hisq_rotation left
                # behind. Hooking it would rotate twice.
                module = resolve_module(layer, name)
                if not isinstance(module, nn.Linear):
                    continue

                def rotate_input(
                    _module: nn.Module,
                    args: tuple[torch.Tensor, ...],
                    selected_rotation: HiSQRotation = rotation,
                ) -> tuple[torch.Tensor, ...]:
                    return (
                        apply_hisq_rotation(args[0], selected_rotation),
                        *args[1:],
                    )

                installed.append(module.register_forward_pre_hook(rotate_input))
            return installed

        hessian_sample_weight = 1.0

        def install_collect_hooks(names: list[str]) -> list[object]:
            installed: list[object] = []
            for name in names:
                quantizer = quantizers[name]

                def collect(
                    _module: nn.Module,
                    args: tuple[torch.Tensor, ...],
                    _out: torch.Tensor,
                    q: GPTQ = quantizer,
                    scale: torch.Tensor | None = layer_scales.get(name),
                ) -> None:
                    activations = args[0]
                    if scale is not None:
                        activations = activations * scale.to(
                            device=activations.device,
                            dtype=activations.dtype,
                        )
                    if scale is not None or hisq_rotation_enabled or not getattr(
                        _module, "faquant_input_prequantized", False
                    ):
                        activations = fake_quantize(
                            activations,
                            quant_format=config.quant_format,
                            bits=config.bits,
                            group_size=config.activation_group_size,
                            symmetric=config.symmetric,
                            clip_ratio=config.clip_ratio,
                        )
                    if hessian_sample_weight != 1.0:
                        activations = activations * (hessian_sample_weight**0.5)
                    q.add_batch(activations)

                installed.append(
                    resolve(layer, name).register_forward_hook(collect)
                )
            return installed

        def collect_hessians(names: list[str]) -> None:
            nonlocal hessian_sample_weight
            handles = install_rotation_hooks()
            handles.extend(install_collect_hooks(names))
            attention = layer.self_attn
            original_output_quant = bool(
                getattr(attention, "faquant_attention_output_quant", False)
            )
            original_qk_kwargs = attention.faquant_qk_matmul_quant_kwargs
            original_pv_kwargs = attention.faquant_pv_matmul_quant_kwargs
            base_matmul_kwargs = attention.faquant_matmul_quant_kwargs
            qk_is_mxfp8 = (
                original_qk_kwargs.get("quant_format") == "mxfp8e4m3"
            )
            pv_is_mxfp8 = (
                original_pv_kwargs.get("quant_format") == "mxfp8e4m3"
            )
            mixes_mxfp8 = qk_is_mxfp8 or pv_is_mxfp8
            if smoothquant_enabled or hisq_rotation_enabled:
                attention.faquant_attention_output_quant = False
            try:
                if mxfp8_hessian_mode == "prefix_split":
                    mxfp8_samples = round(
                        len(captured) * mxfp8_hessian_fraction
                    )
                    hessian_sample_weight = 1.0
                    for sample_index, sample in enumerate(captured):
                        use_mxfp8 = sample_index < mxfp8_samples
                        if qk_is_mxfp8:
                            attention.faquant_qk_matmul_quant_kwargs = (
                                original_qk_kwargs
                                if use_mxfp8
                                else base_matmul_kwargs
                            )
                        if pv_is_mxfp8:
                            attention.faquant_pv_matmul_quant_kwargs = (
                                original_pv_kwargs
                                if use_mxfp8
                                else base_matmul_kwargs
                            )
                        layer(sample.unsqueeze(0), **layer_kwargs)
                else:
                    for sample in captured:
                        if mixes_mxfp8 and mxfp8_hessian_fraction < 1.0:
                            attention.faquant_qk_matmul_quant_kwargs = (
                                base_matmul_kwargs
                                if qk_is_mxfp8
                                else original_qk_kwargs
                            )
                            attention.faquant_pv_matmul_quant_kwargs = (
                                base_matmul_kwargs
                                if pv_is_mxfp8
                                else original_pv_kwargs
                            )
                            hessian_sample_weight = 1.0 - mxfp8_hessian_fraction
                            if hessian_sample_weight > 0.0:
                                layer(sample.unsqueeze(0), **layer_kwargs)
                        if mixes_mxfp8 and mxfp8_hessian_fraction > 0.0:
                            attention.faquant_qk_matmul_quant_kwargs = (
                                original_qk_kwargs
                            )
                            attention.faquant_pv_matmul_quant_kwargs = (
                                original_pv_kwargs
                            )
                            hessian_sample_weight = mxfp8_hessian_fraction
                            layer(sample.unsqueeze(0), **layer_kwargs)
                        elif not mixes_mxfp8:
                            hessian_sample_weight = 1.0
                            layer(sample.unsqueeze(0), **layer_kwargs)
            finally:
                hessian_sample_weight = 1.0
                attention.faquant_attention_output_quant = original_output_quant
                attention.faquant_qk_matmul_quant_kwargs = original_qk_kwargs
                attention.faquant_pv_matmul_quant_kwargs = original_pv_kwargs
                for handle in handles:
                    handle.remove()

        # Per-linear SQ and HiSQ rotation each produce a different effective
        # input for projections that otherwise share one Hessian.
        def group_hook_names(group: tuple[str, ...]) -> list[str]:
            if smoothquant_enabled or hisq_rotation_enabled:
                return list(group)
            return [group[0]]

        if not sequential_groups:
            collect_hessians(
                [name for group in layer_groups for name in group_hook_names(group)]
            )

        for group in layer_groups:
            if sequential_groups:
                # Build this group's Hessian from what the layer emits *now*,
                # with every earlier group already quantized. Collecting all
                # four groups in one pass instead means gate/up never see a
                # quantized o_proj and down_proj never sees quantized gate/up,
                # so their Hessians describe an activation path the deployed
                # model never takes.
                collect_hessians(group_hook_names(group))
            if not smoothquant_enabled and not hisq_rotation_enabled:
                primary = quantizers[group[0]]
                for name in group[1:]:
                    quantizers[name].hessian = primary.hessian
                    quantizers[name].samples = primary.samples
            for name, quantizer in quantizers.items():
                if name not in group:
                    continue
                if smoothquant_enabled:
                    linear = resolve(layer, name)
                    scale = layer_scales[name]
                    quantizer.set_weight_override(
                        linear.weight.detach().float()
                        / scale.to(linear.weight.device).unsqueeze(0)
                    )
                    linear.faquant_input_scale = scale
                if hisq_rotation_enabled:
                    quantizer.set_weight_override(rotated_weight_overrides[name])
                stats[f"model.layers.{layer_index}.{name}"] = quantizer.quantize(
                    quant_format=config.quant_format,
                    bits=config.bits,
                    group_size=config.weight_group_size,
                    symmetric=config.symmetric,
                    damp=damp_overrides.get(
                        name.rsplit(".", 1)[-1],
                        config.gptq_damp,
                    ),
                    capture_hif4_metadata=config.qad_capture_weight_metadata,
                    capture_latent_master=config.qad_capture_latent_master,
                    scale_search_steps=scale_search_steps,
                    act_order_within_group=act_order_within_group,
                )
                if config.gptq_propagate_fake_activations:
                    install_fake_quant(layer, name)

        for sample_index, sample in enumerate(captured):
            outputs[sample_index].copy_(layer(sample.unsqueeze(0), **layer_kwargs)[0])
        captured, outputs = outputs, captured
        LOGGER.info("Qwen GPTQ calibrated layer %d/%d", layer_index + 1, len(layers))
    if smoothquant_enabled:
        model.faquant_smoothquant_stats = smoothquant_report
    if hisq_rotation_enabled:
        model.faquant_hisq_rotation_stats = hisq_rotation_report
    return stats
