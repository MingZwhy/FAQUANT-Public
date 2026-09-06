import torch

from faquant.rotation import (
    chunked_generalized_hadamard_transform,
    generalized_hadamard_transform,
    hadamard_transform,
)


def test_hadamard_is_orthogonal_and_self_inverse() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 16)
    rotated = hadamard_transform(x)
    restored = hadamard_transform(rotated)
    torch.testing.assert_close(restored, x, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


def test_generalized_hadamard_12_is_orthogonal() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 12)
    rotated = generalized_hadamard_transform(x)
    torch.testing.assert_close(
        rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


def test_generalized_hadamard_12_preserves_linear_output() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 24)
    weight = torch.randn(7, 24)
    rotated_x = generalized_hadamard_transform(x)
    rotated_weight = generalized_hadamard_transform(weight)
    torch.testing.assert_close(
        rotated_x @ rotated_weight.T,
        x @ weight.T,
        atol=2e-5,
        rtol=2e-5,
    )


def test_chunked_generalized_hadamard_matches_full_transform() -> None:
    torch.manual_seed(0)
    x = torch.randn(11, 24)
    torch.testing.assert_close(
        chunked_generalized_hadamard_transform(x, chunk_rows=3),
        generalized_hadamard_transform(x),
    )
