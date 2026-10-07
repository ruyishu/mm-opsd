import json

import pytest
import torch
from PIL import Image, ImageDraw
from peft import LoraConfig, get_peft_model
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLConfig,
    Qwen3VLForConditionalGeneration, Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from mm_opsd.config import load_config
from mm_opsd.modeling import LANGUAGE_LORA_TARGETS


@pytest.fixture
def tiny_setup(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    vocabulary = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>",
               "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for token in special:
        vocabulary[token] = len(vocabulary)
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.txt"
    vocab_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = Qwen2TokenizerFast(
        vocab_file=str(vocab_path), merges_file=str(merges_path),
        unk_token="<|endoftext|>", pad_token="<|endoftext|>", eos_token="<|im_end|>",
        additional_special_tokens=special[1:],
    )
    template = (
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' }}"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=4096, max_pixels=4096,
        ),
        video_processor=Qwen3VLVideoProcessor(), chat_template=template,
    )
    model_config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3],
                             "mrope_interleaved": True},
        },
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 32, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=vocabulary["<|image_pad|>"], video_token_id=vocabulary["<|video_pad|>"],
        vision_start_token_id=vocabulary["<|vision_start|>"],
        vision_end_token_id=vocabulary["<|vision_end|>"],
        tie_word_embeddings=True,
    )
    model_config._attn_implementation = "sdpa"
    base = Qwen3VLForConditionalGeneration(model_config)
    base.generation_config.eos_token_id = tokenizer.eos_token_id
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0, target_modules=LANGUAGE_LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).polygon([(8, 50), (32, 8), (55, 50)], outline="black")
    image.save(tmp_path / "triangle.png")
    row = {"id": "tiny:0", "question": "Find the angle x.", "answer": "987654",
           "image": "triangle.png", "choices": [], "answer_type": "math", "precision": None}
    config = load_config("configs/smoke.yaml")
    config["data"]["root"] = str(tmp_path)
    config["train"].update(output_dir=str(tmp_path / "run"), max_completion_length=2,
                          max_sequence_length=1024, learning_rate=0.01, max_steps=2,
                          gradient_accumulation_steps=1, save_steps=1, validation_samples=0)
    return model, processor, row, config, tmp_path
