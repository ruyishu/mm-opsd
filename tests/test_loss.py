import pytest
import torch

from mm_opsd.loss import chunked_distillation_loss, forward_kl


@pytest.mark.parametrize("clip", [None, 0.05])
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_chunked_full_vocabulary_loss_and_gradients(clip, chunk_size):
    torch.manual_seed(11)
    head = torch.nn.Linear(7, 31, bias=False).requires_grad_(False)
    dense_states = torch.randn(1, 8, 7, requires_grad=True)
    states = dense_states.detach().clone().requires_grad_(True)
    teacher = torch.randn(1, 8, 7, requires_grad=True)
    dense = forward_kl(head(dense_states), head(teacher), 1.1, clip).mean()
    dense.backward()
    chunked = chunked_distillation_loss(states, teacher, head, chunk_size, 1.1, clip)
    chunked.backward()
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(states.grad, dense_states.grad)
    assert teacher.grad is None
    assert head.weight.grad is None


def test_clip_applies_per_vocabulary_entry():
    student = torch.tensor([[[8.0, -8.0, -8.0]]])
    teacher = torch.tensor([[[-8.0, 8.0, 8.0]]])
    clipped = forward_kl(student, teacher, token_clip=0.05)
    assert clipped.item() > 0.05
