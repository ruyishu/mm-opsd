import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def forward_kl(student_logits, teacher_logits, temperature=1.0, token_clip=0.05):
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    entries = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="none")
    if token_clip is not None:
        entries = entries.clamp(max=token_clip)
    return entries.sum(dim=-1)


def chunked_distillation_loss(
    student_hidden, teacher_hidden, lm_head, chunk_size=32, temperature=1.1, token_clip=0.05,
):
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("Teacher and student must score the same number of completion tokens.")
    if student_hidden.shape[1] == 0:
        raise ValueError("Cannot distill an empty completion.")

    def chunk_loss(student, teacher):
        logits = lm_head(student)
        with torch.no_grad():
            target = lm_head(teacher)
        return forward_kl(logits, target, temperature, token_clip).sum()

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden.shape[1], chunk_size):
        student = student_hidden[:, start:start + chunk_size]
        teacher = teacher_hidden[:, start:start + chunk_size].detach()
        if torch.is_grad_enabled() and student.requires_grad:
            total = total + checkpoint(chunk_loss, student, teacher, use_reentrant=False)
        else:
            total = total + chunk_loss(student, teacher)
    return total / (student_hidden.shape[0] * student_hidden.shape[1])


def chunked_sft_loss(hidden, labels, lm_head, chunk_size=32):
    def chunk_loss(states, targets):
        logits = lm_head(states).float()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        states, targets = hidden[:, start:start + chunk_size], labels[:, start:start + chunk_size]
        total = total + checkpoint(chunk_loss, states, targets, use_reentrant=False)
    return total / labels.numel()
