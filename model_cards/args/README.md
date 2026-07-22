---
license: other
license_name: framenet-academic
license_link: https://framenet.icsi.berkeley.edu/framenet_data
language:
- en
library_name: transformers
pipeline_tag: token-classification
tags:
- frame-semantics
- framenet
- semantic-parsing
- srl
- argument-extraction
- english
base_model: microsoft/deberta-v3-large
---

# texture-frames · argument-extraction head

The **argument-extraction** stage of
[`texture-frames`](https://github.com/texturejc/Texture_Frames), a fast FrameNet
semantic-frame parser. Given a sentence with a marked trigger and its frame, it
finds the spans that fill the frame's roles (frame elements) and labels each.

It fine-tunes [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
on **FrameNet 1.7** with a **detect-then-classify** design — two heads on one
backbone, a single forward pass:

- **Head A — span detection:** a role-agnostic 3-class BIO tagger (`O`/`B`/`I`),
  "is this token part of *an* argument?". Dense signal, arbitrary-length spans.
- **Head B — role classification:** for each detected span, pool its tokens
  (`start ⊕ end ⊕ mean`) and classify into **only the current frame's frame
  elements** (plus a `NULL` reject class), masked via the lexicon.

The input carries the predicate marker and the frame's FE menu
(`{frame} [FE1; FE2; …] : … <t> {trigger} </t> …`). A **`NULL`-bias** at inference
sets the precision/recall operating point.

> This is one of three stages. Use it through the package rather than alone.

## Usage

```bash
pip install git+https://github.com/texturejc/Texture_Frames
```

```python
from texture_frames import FrameParser
parser = FrameParser()
for ann in parser.parse("The chef gave food to the customer ."):
    print([(a.role, a.text) for a in ann.arguments])
# [('Donor', 'The chef'), ('Theme', 'food'), ('Recipient', 'to the customer')]
```

## Files

| File | What |
| ---- | ---- |
| `args2_model.pt` | model `state_dict` (backbone + detection + role heads) |
| `role2id.json` | `{role name → id}` label map (incl. `<NULL>`) + `base_model` |
| tokenizer files | DeBERTa-v3 tokenizer with the `<t>` / `</t>` markers added |

Loading is handled by `texture_frames.weights.load_args`.

## Results

Open-Sesame test split, weighted F1 (non-core FEs = 0.5):

| Metric | This head | T5 baseline |
| ------ | --------- | ----------- |
| Argument F1 | **0.750** | 0.753 |
| Speed | single forward pass (~50–60 ms) | 3 beam-search passes |

Parity with the generative baseline while running ~4× faster. The encoder went
0.628 (flat BIO) → 0.712 (detect-then-classify) → 0.750 (+ WordNet augmentation)
across redesigns.

## Training

`microsoft/deberta-v3-large`, AdamW lr 1e-5, warmup 0.06, weight decay 0.01,
batch 16, max length 320, bf16, 6 epochs with WordNet synonym augmentation.
Data: FrameNet 1.7 (NLTK), Open-Sesame splits.

## Licence

**Code (the package): MIT.** **Weights:** trained on **FrameNet 1.7**, which
carries its own academic-use terms — review them before redistributing.

## Citation

```bibtex
@software{texture_frames,
  author = {Carney, James},
  title  = {texture-frames: a fast DeBERTa encoder FrameNet parser},
  url    = {https://github.com/texturejc/Texture_Frames},
  year   = {2026}
}
```

Builds on David Chanin's
[`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer);
thanks to the Berkeley FrameNet and Open-Sesame projects.
