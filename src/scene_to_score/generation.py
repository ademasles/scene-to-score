"""MusicLDM conditioned on a CLAP vector instead of a text prompt.

MusicLDM conditions its UNet on the 512-d CLAP embedding of the prompt. Text and audio CLAP
embeddings share one space, so a CLAP *audio* vector, real or predicted, can stand in for the
prompt through ``prompt_embeds``. With transformers 5, ``ClapModel.get_text_features`` returns an
output object rather than a tensor, which the pipeline does not expect: we therefore always pass
``prompt_embeds`` and ``negative_prompt_embeds`` ourselves (the latter is the empty text).

The CLAP used here is the pipeline's own text encoder, so targets computed with
:func:`audio_embedding` live in exactly the space the generator was trained on.
"""

from collections.abc import Sequence

import numpy as np
import torch
from diffusers.pipelines.deprecated.musicldm.pipeline_musicldm import MusicLDMPipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput
from transformers import ClapFeatureExtractor, ClapModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPooling

MODEL_ID = "ucsd-reach/musicldm"


def load_pipeline(
    model_id: str = MODEL_ID, device: torch.device | None = None, half: bool = True
) -> MusicLDMPipeline:
    device = device or torch.device("cpu")
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    pipe = MusicLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if pipe.feature_extractor is None:
        raise ValueError(f"{model_id} ships no CLAP feature extractor")
    return pipe.to(device)


@torch.inference_mode()
def audio_embedding(
    model: ClapModel, feature_extractor: ClapFeatureExtractor, waves: Sequence[np.ndarray]
) -> torch.Tensor:
    """Unit-norm CLAP audio vectors (B x 512, float32) of mono waves at the extractor's rate."""
    inputs = feature_extractor(
        list(waves), sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
    )
    out = model.get_audio_features(
        input_features=inputs["input_features"].to(model.device, model.dtype),
        is_longer=inputs["is_longer"].to(model.device),
    )
    return _projected(out)


@torch.inference_mode()
def empty_text_embedding(model: ClapModel, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """CLAP vector of the empty prompt (1 x 512), the unconditional branch of guidance."""
    tokens = tokenizer([""], padding=True, return_tensors="pt")
    out = model.get_text_features(
        input_ids=tokens["input_ids"].to(model.device),
        attention_mask=tokens["attention_mask"].to(model.device),
    )
    return _projected(out)


def _projected(out: object) -> torch.Tensor:
    """transformers 5 puts the normalised projection in ``pooler_output``."""
    if not isinstance(out, BaseModelOutputWithPooling) or out.pooler_output is None:
        raise TypeError(f"expected BaseModelOutputWithPooling, got {type(out).__name__}")
    return out.pooler_output.float()


def generate(
    pipe: MusicLDMPipeline,
    z: torch.Tensor,
    negative: torch.Tensor,
    *,
    seconds: float = 10.0,
    steps: int = 200,
    guidance: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """One waveform per row of ``z`` (B x 512), at the vocoder's rate (B x samples)."""
    if z.ndim != 2 or negative.shape != (1, z.shape[1]):
        raise ValueError(f"z must be (B, d) and negative (1, d), got {z.shape}, {negative.shape}")
    dtype = pipe.text_encoder.dtype
    out = pipe(
        prompt_embeds=z.to(pipe.device, dtype),
        negative_prompt_embeds=negative.expand_as(z).to(pipe.device, dtype),
        audio_length_in_s=seconds,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        output_type="np",
    )
    audios = out.audios if isinstance(out, AudioPipelineOutput) else out[0]
    return np.asarray(audios, dtype=np.float32)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b``."""
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    return (a @ b.T).cpu().numpy()


def round_trip_report(similarity: np.ndarray) -> dict[str, float]:
    """Summarise CLAP(generated audio) x conditioning vectors, matched on the diagonal.

    A generator that follows its conditioning gives each output a higher similarity to its
    own vector than to the others, so its own vector ranks first.
    """
    n = similarity.shape[0]
    if similarity.shape != (n, n) or n < 2:
        raise ValueError(f"need a square matrix of at least 2 x 2, got {similarity.shape}")
    own = np.diag(similarity)
    others = similarity[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    ranks = 1 + (others > own[:, None]).sum(axis=1)
    return {
        "own_cosine": float(own.mean()),
        "other_cosine": float(others.mean()),
        "mean_rank": float(ranks.mean()),
        "top1": float((ranks == 1).mean()),
    }
