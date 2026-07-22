---
license: other
license_name: framenet-academic
license_link: https://framenet.icsi.berkeley.edu/framenet_data
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- frame-semantics
- framenet
- semantic-parsing
- srl
- english
base_model: microsoft/deberta-v3-large
---

# texture-frames · frame-classification head

The **frame-classification** stage of
[`texture-frames`](https://github.com/texturejc/Texture_Frames), a fast FrameNet
semantic-frame parser. Given a sentence with a marked trigger, it predicts which
of ~1,221 FrameNet frames the trigger evokes.

It fine-tunes [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
on **FrameNet 1.7** and uses **marker-token pooling**: the trigger is wrapped in
entity markers (`… <t> gave </t> …`) and the frame representation is the
concatenation of the two marker tokens' hidden states (not `[CLS]`), focusing the
classifier on the predicate. A single forward pass — no beam search.

> This is one of three stages. Use it through the package rather than alone.

## Usage

```bash
pip install git+https://github.com/texturejc/Texture_Frames
```

```python
from texture_frames import FrameParser
parser = FrameParser()
for ann in parser.parse("The chef gave food to the customer ."):
    print(ann.trigger, "->", ann.frame)
# gave -> Giving
```

At inference the logits are **soft-masked** toward the trigger's candidate frames
(from the FrameNet lexicon) so a confident non-candidate can still win.

## Files

| File | What |
| ---- | ---- |
| `frame2_model.pt` | model `state_dict` (backbone + marker-pooling classifier) |
| `frame2id.json` | `{frame name → id}` label map + `base_model` |
| tokenizer files | DeBERTa-v3 tokenizer with the `<t>` / `</t>` markers added |

Loading is handled by `texture_frames.weights.load_frame`.

## Results

Open-Sesame test split:

| Metric | This head | T5 baseline |
| ------ | --------- | ----------- |
| Frame accuracy | **0.863–0.868** | 0.887 |
| Speed | single forward pass (~50–60 ms) | 3 beam-search passes |

Competitive (~−0.02); the residual gap is largely a candidate-lexicon coverage
ceiling (2.2% of gold frames fall outside the candidate set), not discrimination.

## Training

`microsoft/deberta-v3-large`, AdamW lr 1e-5, warmup 0.06, weight decay 0.01,
batch 16, max length 320, bf16, 5 epochs. Data: FrameNet 1.7 (NLTK), Open-Sesame
splits.

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
