import torch

from mm_opsd.loss import chunked_distillation_loss
from mm_opsd.modeling import (
    attach_completion, base_model, completion_hidden, encode_prompt, generation_kwargs,
)


def test_real_multimodal_teacher_alignment_and_freezing(tiny_setup):
    model, processor, row, config, root = tiny_setup
    device = torch.device("cpu")
    student = encode_prompt(processor, row, root, device)
    teacher = encode_prompt(processor, row, root, device, teacher=True)
    assert row["answer"] not in processor.tokenizer.decode(student["input_ids"][0])
    assert row["answer"] in processor.tokenizer.decode(teacher["input_ids"][0])
    torch.testing.assert_close(student["pixel_values"], teacher["pixel_values"])
    assert student["input_ids"].shape[1] != teacher["input_ids"].shape[1]
    completion = torch.tensor([processor.tokenizer.encode("12", add_special_tokens=False)])
    model.train()
    base = base_model(model)
    base.visual.eval()
    states = completion_hidden(model, student, completion)
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        teacher_before = completion_hidden(model, teacher, completion).clone()
    model.train()
    base.visual.eval()
    loss = chunked_distillation_loss(states, teacher_before, base.lm_head, chunk_size=1)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "lora_" in name)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.visual.parameters())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        teacher_after = completion_hidden(model, teacher, completion)
    torch.testing.assert_close(teacher_before, teacher_after, rtol=0, atol=0)
    with torch.no_grad():
        actual = completion_hidden(model, student, completion)
        full = attach_completion(student, completion)
        positions, _ = base.model.get_rope_index(full["input_ids"], full["image_grid_thw"],
                                                attention_mask=full["attention_mask"])
        direct = model(**full, position_ids=positions, use_cache=False).logits
        start = student["input_ids"].shape[1] - 1
        torch.testing.assert_close(base.lm_head(actual), direct[:, start:start + completion.shape[1]])


def test_real_model_generation_accepts_image(tiny_setup):
    model, processor, row, config, root = tiny_setup
    model.eval()
    prompt = encode_prompt(processor, row, root, torch.device("cpu"))
    with torch.no_grad():
        generated = model.generate(**prompt, **generation_kwargs(
            model, 2, sample=True, train_config=config["train"], pad_token_id=processor.tokenizer.pad_token_id,
        ))
    assert generated.shape[1] > prompt["input_ids"].shape[1]
