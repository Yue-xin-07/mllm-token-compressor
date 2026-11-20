import pytest
import torch

from compressor.token_pruner import TokenPruner


def sample(device="cpu", dtype=torch.float32):
    hidden = torch.arange(32, device=device, dtype=dtype).reshape(8, 4)
    positions = torch.tensor([
        [0, 1, 2, 2, 2, 2, 8, 9],
        [0, 1, 2, 2, 3, 3, 8, 9],
        [0, 1, 2, 3, 2, 3, 8, 9],
    ], device=device)
    mask = torch.tensor([False, False, True, True, True, True, False, False],
                        device=device)
    weights = torch.ones(2, 3, 4, device=device, dtype=dtype)
    return hidden, positions, mask, weights


def test_gathers_positions_text_and_hidden_in_original_order():
    inputs = sample()
    originals = [x.clone() for x in inputs]
    result = TokenPruner()(*inputs)
    expected = torch.tensor([0, 1, 2, 3, 4, 6, 7])
    torch.testing.assert_close(result.kept_indices, expected)
    torch.testing.assert_close(result.hidden_states, inputs[0][expected])
    torch.testing.assert_close(result.position_ids, inputs[1][:, expected])
    assert result.visual_keep_count == 3
    assert result.next_position == 10  # compressed length is 7
    for actual, original in zip(inputs, originals):
        torch.testing.assert_close(actual, original)


def test_selects_highest_score_and_preserves_original_coordinates():
    hidden, positions, mask, _ = sample()
    attention = torch.tensor([[[0., 0., 0., 1.]]])
    result = TokenPruner()(hidden, positions, mask, attention)
    assert result.kept_indices.tolist() == [0, 1, 5, 6, 7]
    assert result.position_ids[:, 2].tolist() == [2, 3, 3]


@pytest.mark.parametrize("query_count", [0, 2])
def test_missing_evidence_retains_everything(query_count):
    hidden, positions, mask, _ = sample()
    result = TokenPruner()(hidden, positions, mask, torch.zeros(1, query_count, 4))
    assert result.visual_keep_count == 4
    torch.testing.assert_close(result.hidden_states, hidden)


def test_no_visual_tokens():
    hidden, positions, mask, _ = sample()
    mask.zero_()
    result = TokenPruner()(hidden, positions, mask, torch.empty(2, 3, 0))
    torch.testing.assert_close(result.hidden_states, hidden)
    assert result.visual_keep_count == 0


def test_empty_sequence():
    result = TokenPruner()(torch.empty(0, 4), torch.empty(3, 0, dtype=torch.long),
                           torch.empty(0, dtype=torch.bool), torch.empty(1, 0, 0))
    assert result.hidden_states.shape == (0, 4)
    assert result.next_position == 0


def test_single_visual_and_all_visual():
    result = TokenPruner()(torch.ones(1, 4), torch.tensor([[4], [8], [3]]),
                           torch.ones(1, dtype=torch.bool), torch.ones(1, 1, 1))
    assert result.kept_indices.tolist() == [0]
    assert result.next_position == 9


def test_keep_all_config_and_no_autograd():
    hidden, positions, mask, attention = sample()
    hidden.requires_grad_()
    result = TokenPruner(1, 1)(hidden, positions, mask, attention)
    torch.testing.assert_close(result.hidden_states, hidden)
    assert not result.hidden_states.requires_grad


@pytest.mark.parametrize("index,replacement", [
    (0, torch.ones(8, dtype=torch.float32)),
    (0, torch.ones(8, 4, dtype=torch.long)),
    (1, torch.ones(2, 8, dtype=torch.long)),
    (1, torch.ones(3, 8, dtype=torch.float32)),
    (1, -torch.ones(3, 8, dtype=torch.long)),
    (2, torch.ones(8, dtype=torch.long)),
    (2, torch.ones(7, dtype=torch.bool)),
    (3, torch.ones(1, 1, 3)),
    (3, torch.ones(1, 4)),
    (3, -torch.ones(1, 1, 4)),
])
def test_invalid_input(index, replacement):
    inputs = list(sample())
    inputs[index] = replacement
    with pytest.raises(ValueError):
        TokenPruner()(*inputs)


@pytest.mark.parametrize("ratios", [(0, 1), (.8, .2), (.1, 2), (float("nan"), 1)])
def test_invalid_configuration(ratios):
    with pytest.raises(ValueError):
        TokenPruner(*ratios)


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["uniform", "concentrated", "varied"])
def test_cuda_matches_cpu(dtype, kind):
    inputs = list(sample(dtype=dtype))
    if kind == "concentrated":
        inputs[3] = torch.tensor([[[0., 0., 0., 1.]]], dtype=dtype)
    elif kind == "varied":
        inputs[3] = torch.tensor([[[1., 2., 4., 8.], [8., 4., 2., 1.]]],
                                 dtype=dtype)
    pruner = TokenPruner()
    cpu = pruner(*inputs)
    gpu = pruner(*(x.cuda() for x in inputs))
    torch.testing.assert_close(gpu.kept_indices.cpu(), cpu.kept_indices)
    torch.testing.assert_close(gpu.hidden_states.cpu(), cpu.hidden_states)
    torch.testing.assert_close(gpu.position_ids.cpu(), cpu.position_ids)
    assert gpu.entropy == pytest.approx(cpu.entropy, abs=1e-6)
    assert gpu.visual_keep_count == cpu.visual_keep_count
    assert gpu.next_position == cpu.next_position
    assert gpu.hidden_states.dtype == dtype


@CUDA_REQUIRED
def test_cuda_rejects_mixed_devices():
    inputs = list(sample())
    inputs[0] = inputs[0].cuda()
    with pytest.raises(ValueError, match="same device"):
        TokenPruner()(*inputs)


@CUDA_REQUIRED
def test_cuda_repeated_calls_release_allocated_memory():
    import gc

    device = "cuda"
    hidden = torch.randn(2048, 128, device=device, dtype=torch.bfloat16)
    positions = torch.arange(2048, device=device).expand(3, -1)
    mask = torch.arange(2048, device=device) % 2 == 0
    attention = torch.rand(4, 16, 1024, device=device, dtype=torch.bfloat16)
    pruner = TokenPruner()

    def run(count):
        for _ in range(count):
            output = pruner(hidden, positions, mask, attention)
            assert not output.hidden_states.requires_grad
            del output
        gc.collect()
        torch.cuda.synchronize()

    run(10)  # initialize kernels and allocator before measuring
    baseline = torch.cuda.memory_allocated()
    allocated, reserved = [], []
    for _ in range(5):
        run(30)
        allocated.append(torch.cuda.memory_allocated())
        reserved.append(torch.cuda.memory_reserved())
    print(f"CUDA memory bytes: baseline={baseline}, allocated={allocated}, reserved={reserved}")
    # Cached blocks may remain reserved; live tensor allocations must be released.
    assert all(value <= baseline + 4096 for value in allocated)


def test_removed_maximum_coordinate_does_not_shift_next_position():
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    positions = torch.tensor([[0, 1, 2, 3, 100],
                              [0, 4, 5, 6, 80],
                              [0, 7, 8, 9, 90]])
    mask = torch.tensor([False, True, True, True, True])
    result = TokenPruner()(hidden, positions, mask,
                           torch.tensor([[[1., 0., 0., 0.]]]))
    assert result.kept_indices.tolist() == [0, 1]
    assert result.next_position == 101
    assert int(result.position_ids.max()) == 7


def test_multiple_visual_spans_and_noncontiguous_inputs():
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8).T
    positions = torch.arange(24).reshape(8, 3).T
    mask = torch.tensor([False, True, True, False, False, True, True, False])
    result = TokenPruner(.5, .5)(
        hidden, positions, mask, torch.tensor([[[1., 4., 3., 2.]]]))
    expected = torch.tensor([0, 2, 3, 4, 5, 7])
    torch.testing.assert_close(result.kept_indices, expected)
    torch.testing.assert_close(result.hidden_states, hidden[expected])
    torch.testing.assert_close(result.position_ids, positions[:, expected])
