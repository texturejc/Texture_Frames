"""End-to-end plumbing smoke test for the args v2 training/eval pipeline on a tiny
random backbone + fake data (no download, no nltk). Verifies the custom collator,
the Args2Trainer (train step + eval loss via prediction_step), and the eval decode
path. Skips cleanly if torch/transformers are unavailable."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from transformers import AutoConfig, AutoModel, TrainingArguments
except Exception:  # pragma: no cover
    print("torch/transformers unavailable — skipping"); sys.exit(0)

from args2_data import decode_detect_spans  # noqa: E402
from model_args2 import Args2Model  # noqa: E402
from train_args2 import Args2Collator, Args2Trainer  # noqa: E402


class _StubTok:
    pad_token_id = 0


def _tiny_model(num_roles=5):
    cfg = AutoConfig.for_model(
        "deberta-v2", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, vocab_size=100, max_position_embeddings=64,
    )
    return Args2Model(AutoModel.from_config(cfg), num_roles=num_roles)


def _rows(n, length=8):
    return [
        {
            "input_ids": [1] * length,
            "attention_mask": [1] * length,
            "detect_labels": [-100, 0, 1, 2, 0, 0, 0, -100][:length],
            "spans": [(2, 3, 1), (4, 4, 0)],
        }
        for _ in range(n)
    ]


def test_collator_pads_and_preserves_spans():
    coll = Args2Collator(_StubTok())
    feats = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
         "detect_labels": [0, 1, 2], "spans": [(1, 2, 3)]},
        {"input_ids": [4, 5], "attention_mask": [1, 1],
         "detect_labels": [0, 1], "spans": []},
    ]
    b = coll(feats)
    assert b["input_ids"].shape == (2, 3)
    assert b["input_ids"][1].tolist() == [4, 5, 0]           # right-padded
    assert b["detect_labels"][1].tolist() == [0, 1, -100]     # label pad = ignore
    assert b["spans"] == [[(1, 2, 3)], []]                    # ragged spans intact


def test_trainer_train_step_and_eval_loss():
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, max_steps=2, per_device_train_batch_size=2,
            eval_strategy="no", save_strategy="no", report_to="none", use_cpu=True,
            remove_unused_columns=False, label_names=["detect_labels", "spans"],
            logging_strategy="no",
        )
        trainer = Args2Trainer(
            model=model, args=args,
            train_dataset=_rows(4), eval_dataset=_rows(2),
            data_collator=Args2Collator(_StubTok()),
        )
        trainer.train()                       # train step: compute_loss + collator
        metrics = trainer.evaluate()          # prediction_step -> eval loss
        assert "eval_loss" in metrics and metrics["eval_loss"] == metrics["eval_loss"]


def test_eval_decode_path_runs():
    # detection argmax -> spans -> role head -> role argmax, end to end on tiny model
    model = _tiny_model(num_roles=5)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 8))
    attention_mask = torch.ones(1, 8, dtype=torch.long)
    offsets = [(0, 0), (0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17), (0, 0)]
    prefix_len = 5
    with torch.no_grad():
        hidden, detect_logits = model.encode(input_ids, attention_mask)
        detect_pred = detect_logits[0].argmax(-1).tolist()
        spans = decode_detect_spans(offsets, detect_pred, prefix_len)
        if spans:
            span_bi = [(0, s, e) for (s, e, _, _) in spans]
            role_logits = model.role_logits_for_spans(hidden, span_bi)
            assert role_logits.shape == (len(spans), 5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
