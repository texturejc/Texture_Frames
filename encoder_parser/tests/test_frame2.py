"""Tests for frame v2: pure marker-position finding + model/collator/trainer
plumbing on a tiny random backbone (no download). Skips if torch unavailable."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame2_data import find_marker_positions  # noqa: E402

START, END = 50, 51  # pretend marker ids


def test_find_marker_positions_basic():
    ids = [1, 2, START, 3, 4, END, 5]
    assert find_marker_positions(ids, START, END) == (2, 5)


def test_find_marker_positions_fallback_when_truncated():
    ids = [1, 2, START, 3, 4]          # </t> truncated away
    assert find_marker_positions(ids, START, END) == (2, 4)   # end -> last token
    ids2 = [1, 2, 3]                    # neither marker present
    assert find_marker_positions(ids2, START, END) == (0, 2)  # CLS / last


try:
    import torch
    from transformers import AutoConfig, AutoModel, TrainingArguments
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

if _HAVE_TORCH:
    from model_frame2 import FrameMarkerModel
    from train_frame2 import FrameMarkerCollator

    class _StubTok:
        pad_token_id = 0

    def _tiny_model(num_frames=7):
        cfg = AutoConfig.for_model(
            "deberta-v2", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, vocab_size=100, max_position_embeddings=64,
        )
        return FrameMarkerModel(AutoModel.from_config(cfg), num_frames=num_frames)

    def test_forward_loss_and_shapes():
        model = _tiny_model()
        input_ids = torch.randint(0, 100, (2, 8))
        attn = torch.ones(2, 8, dtype=torch.long)
        start_pos = torch.tensor([2, 3])
        end_pos = torch.tensor([5, 6])
        labels = torch.tensor([1, 4])
        out = model(input_ids, attn, start_pos, end_pos, labels=labels)
        assert out["logits"].shape == (2, 7)
        assert out["loss"].dim() == 0
        out["loss"].backward()
        assert model.classifier[0].weight.grad is not None

    def test_collator_stacks_positions_and_pads():
        coll = FrameMarkerCollator(_StubTok())
        feats = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "start_pos": 1, "end_pos": 2, "labels": 3},
            {"input_ids": [4, 5], "attention_mask": [1, 1],
             "start_pos": 0, "end_pos": 1, "labels": 5},
        ]
        b = coll(feats)
        assert b["input_ids"].shape == (2, 3)
        assert b["input_ids"][1].tolist() == [4, 5, 0]
        assert b["start_pos"].tolist() == [1, 0] and b["end_pos"].tolist() == [2, 1]
        assert b["labels"].tolist() == [3, 5]

    def test_trainer_one_step():
        from transformers import Trainer
        model = _tiny_model()
        rows = [{"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1],
                 "start_pos": 1, "end_pos": 2, "labels": i % 7} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, max_steps=2, per_device_train_batch_size=2,
                eval_strategy="no", save_strategy="no", report_to="none",
                use_cpu=True, remove_unused_columns=False, logging_strategy="no",
            )
            Trainer(model=model, args=args, train_dataset=rows,
                    data_collator=FrameMarkerCollator(_StubTok())).train()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
