# Public release manifest

## Included

- Qwen3/HiF4 numerical implementation required by the final recipe
- HiSQ and partially offline Linear input rotations for Qwen3
- PTQ, Smooth-QK calibration, QAD, protection materialization, and evaluation workflows
- Qwen3-32B MMLU shard list and shard merge workflow
- Wan2.2 HiF4 Linear and attention implementation, including the Triton QK/PV kernel
- Wan2.2 video generation, VBench evaluation, paired comparison, GPTQ, and Smooth-QK workflows
- Official aggregate MMLU and LongBench scores for Qwen3-8B and Qwen3-32B
- Aggregate VBench scores and paired comparisons for Wan2.2-T2V-A14B
- Synthetic/unit tests
- Public dependency pins, licenses, and citations

## Excluded

- Model weights and checkpoints
- Calibration tensors and other binary artifacts
- Training and benchmark datasets
- Per-sample prompts, labels, model generations, and logs
- Generated videos and per-video scores
- Code, prompts, and artifacts of third-party projects, which the workflows read from user-provided checkouts
- Private experiment history and failed sweeps
- Cluster orchestration and infrastructure configuration
- Internal filesystem paths, hosts, credentials, and organization identifiers
- Private repository history, branches, remotes, and author metadata

## Artifact identity

The five expected checkpoint SHA-256 values are published in
`results/final_metrics.json`. The separately distributed
[Hugging Face checkpoint](https://huggingface.co/MingZwhy/qwen3-8b-hif4-l16-17-18)
must match all five values.
