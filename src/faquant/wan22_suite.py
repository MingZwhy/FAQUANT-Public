"""VBench standard prompt suite and the generation protocol for Wan2.2-T2V-A14B.

The protocol follows the VBench team's own Wan2.2-T2V-A14B entries
(sampled 2026-02-18, Wan2.2 commit 81e6735): 1280x720, 81 frames, 16 fps,
the T2V-A14B defaults (40 UniPC steps, shift 12, guidance 3.0 low / 4.0 high
noise, the default negative prompt) and no prompt extension. VBench samples
five videos per prompt and 25 for temporal flickering, each with its own seed;
the seed here is a hash of the prompt and the sample index, so it does not
depend on job order or the number of workers.

Two prompts appear twice in VBench_full_info.json with different dimensions;
standard-mode evaluation looks both up by the same file name, so they are
generated once: 6,220 videos.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SEED_SALT = "faquant-wan22-vbench-v1"

DIMENSIONS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
)


@dataclass(frozen=True)
class Protocol:
    task: str = "t2v-A14B"
    size: str = "1280*720"
    frame_num: int = 81
    sample_steps: int = 40
    sample_shift: float = 12.0
    guide_scale: tuple[float, float] = (3.0, 4.0)
    sample_solver: str = "unipc"
    fps: int = 16

    def to_json(self) -> dict:
        return asdict(self)


OFFICIAL = Protocol()


def video_seed(prompt: str, index: int) -> int:
    digest = hashlib.sha256(f"{SEED_SALT}|{prompt}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def video_name(prompt: str, index: int) -> str:
    return f"{prompt}-{index}.mp4"


def samples_per_prompt(dimensions: list[str]) -> int:
    return 25 if "temporal_flickering" in dimensions else 5


def suite_jobs(full_info: Path | str) -> list[dict]:
    """One job per unique (prompt, index), ordered by sample index first.

    VBench_full_info lists the 75 temporal-flickering prompts (25 samples each)
    first; ordering by index makes the first pass cover every prompt once, so a
    partial run already scores every dimension.
    """

    entries = json.loads(Path(full_info).read_text())
    keyed, seen = [], set()
    for position, entry in enumerate(entries):
        prompt = entry["prompt_en"]
        for index in range(samples_per_prompt(entry["dimension"])):
            if (prompt, index) in seen:
                continue
            seen.add((prompt, index))
            keyed.append(((index, position), {
                "prompt": prompt,
                "index": index,
                "seed": video_seed(prompt, index),
                "name": video_name(prompt, index),
            }))
    return [job for _, job in sorted(keyed, key=lambda item: item[0])]


def meta_name(job: dict) -> str:
    key = hashlib.sha1(job["prompt"].encode()).hexdigest()[:16]
    return f"{key}-{job['index']}.json"
