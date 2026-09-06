# Public release manifest

## Included

- Qwen3/HiF4 numerical implementation required by the final recipe
- PTQ, Smooth-QK calibration, QAD, protection materialization, and evaluation workflows
- Official aggregate MMLU and LongBench scores
- Synthetic/unit tests
- Public dependency pins, licenses, and citations

## Excluded

- Model weights and checkpoints
- Calibration tensors and other binary artifacts
- Training and benchmark datasets
- Per-sample prompts, labels, model generations, and logs
- Private experiment history and failed sweeps
- Cluster orchestration and infrastructure configuration
- Internal filesystem paths, hosts, credentials, and organization identifiers
- Private repository history, branches, remotes, and author metadata

## Artifact identity

The five expected checkpoint SHA-256 values are published in
`results/final_metrics.json`. A future separately distributed checkpoint must
match all five values.
