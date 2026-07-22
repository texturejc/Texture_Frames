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
- english
base_model: microsoft/deberta-v3-large
---

# texture-frames · trigger-identification head

The **trigger-identification** stage of
[`texture-frames`](https://github.com/texturejc/Texture_Frames), a fast FrameNet
semantic-frame parser. A per-token classifier (`O` / `TRIGGER`) that finds the
words in a sentence that evoke a frame.

It fine-tunes [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
on **FrameNet 1.7** (via NLTK) with the **Open-Sesame** document splits, scored
with the upstream word-level F1 so it is directly comparable to prior work.

> This is one of three stages. Use it through the package rather than alone; the
> pipeline chains trigger → frame → arguments.

## Usage

```bash
pip install git+https://github.com/texturejc/Texture_Frames
```

```python
from texture_frames import FrameParser
parser = FrameParser()   # downloads all three heads on first use
for ann in parser.parse("The chef gave food to the customer ."):
    print(ann.trigger, "->", ann.frame)
# gave -> Giving
```

A standard `AutoModelForTokenClassification`, so it also loads directly with
`transformers` — but the package handles the word-level alignment (a
`trigger_bias` lever trades precision for recall).

## Results

Open-Sesame test split, word-level F1 (same metric as the T5 baseline):

| Metric | This head | T5 baseline |
| ------ | --------- | ----------- |
| Trigger F1 | **0.750** | 0.735 |
| Speed | single forward pass (~50–60 ms) | 3 beam-search passes |

Ahead of the baseline, and ~3–4× faster (no autoregressive decoding).

## Training

`microsoft/deberta-v3-large`, AdamW lr 1e-5, warmup 0.06, weight decay 0.01,
batch 16, max length 320, bf16, 5 epochs. Data: FrameNet 1.7 (NLTK), Open-Sesame
splits. See the [repo](https://github.com/texturejc/Texture_Frames) for details.

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
