"""Smoke test for Args2Model forward/pooling/loss on a tiny random backbone
(no download). Exercises the two-head plumbing that can't be checked by the pure
data tests. Skips cleanly if torch/transformers aren't installed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from transformers import AutoConfig, AutoModel
except Exception:  # pragma: no cover
    print("torch/transformers unavailable — skipping"); sys.exit(0)

from model_args2 import IGNORE_INDEX, Args2Model  # noqa: E402


def _tiny_model(num_roles=5):
    cfg = AutoConfig.for_model(
        "deberta-v2", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, vocab_size=100, max_position_embeddings=64,
    )
    return Args2Model(AutoModel.from_config(cfg), num_roles=num_roles)


def test_forward_produces_scalar_loss_that_backprops():
    torch.manual_seed(0)
    model = _tiny_model()
    B, T = 2, 8
    input_ids = torch.randint(0, 100, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    detect_labels = torch.tensor([
        [IGNORE_INDEX, 0, 1, 2, 0, 0, 0, IGNORE_INDEX],
        [IGNORE_INDEX, 0, 1, 0, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
    ])
    # per-example spans: (start_tok, end_tok_inclusive, role_id)
    spans = [[(2, 3, 1), (4, 4, 0)], [(2, 2, 2)]]

    out = model(input_ids, attention_mask, detect_labels=detect_labels, spans=spans)
    loss = out["loss"]
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert out["detect_logits"].shape == (B, T, 3)
    loss.backward()  # gradients flow through both heads + backbone
    assert model.role_head[0].weight.grad is not None
    assert model.detect_head.weight.grad is not None


def test_empty_spans_still_gives_finite_loss():
    model = _tiny_model()
    input_ids = torch.randint(0, 100, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    detect_labels = torch.tensor([[IGNORE_INDEX, 0, 0, 0, 0, IGNORE_INDEX]])
    out = model(input_ids, attention_mask, detect_labels=detect_labels, spans=[[]])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()  # must not error with no spans


def test_inference_helpers_shapes():
    model = _tiny_model(num_roles=5)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        hidden, detect_logits = model.encode(input_ids, attention_mask)
        assert detect_logits.shape == (1, 6, 3)
        role_logits = model.role_logits_for_spans(hidden, [(0, 2, 4), (0, 1, 1)])
        assert role_logits.shape == (2, 5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
