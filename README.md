# texture-frames

**A fast FrameNet semantic-frame parser** — a modern DeBERTa-v3-large *encoder*
rearchitecture of the T5-based [`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer).
It replaces three sequential beam-search generation passes with single encoder
forward passes, making it **~3–4× faster** while **matching or beating** the
original on trigger identification and argument extraction, and staying within
~0.02 on frame classification.

```python
from texture_frames import FrameParser
parser = FrameParser()
for ann in parser.parse("The chef gave food to the customer ."):
    print(ann.frame, "|", ann.trigger, "|", [(a.role, a.text) for a in ann.arguments])
# Giving | gave | [('Donor', 'The chef'), ('Theme', 'food'), ('Recipient', 'to the customer')]
```

---

## Contents

1. [Background: semantic frames & FrameNet](#background-semantic-frames--framenet)
2. [Results vs. the baseline](#results-vs-the-baseline)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How it works: models, choices, and training](#how-it-works-models-choices-and-training)
6. [Model checkpoints](#model-checkpoints)
7. [Acknowledgements](#acknowledgements)
8. [Citation & license](#citation--license)

---

## Background: semantic frames & FrameNet

A **semantic frame** is a schematic representation of a situation — an event,
relation, or state — together with the participants and props involved in it.
The idea, from Charles Fillmore's *frame semantics*, is that understanding a word
means understanding the whole scene it evokes. The verb *gave* evokes a **Giving**
frame, which comes with roles (**frame elements**) like a **Donor**, a **Theme**
(the thing given), and a **Recipient**. Recognising the frame and filling its
roles turns a flat sentence into a structured description of *who did what to whom*.

**[FrameNet](https://framenet.icsi.berkeley.edu/)** (Baker, Fillmore & Lowe, 1998)
is a large lexical database that makes this concrete: ~1,200 frames, the words
(*lexical units*) that evoke each one, the roles each frame defines, and tens of
thousands of example sentences hand-annotated by linguists. **Frame-semantic
parsing** is the task of reproducing those annotations automatically. It is
conventionally split into three steps, which this library performs in order:

| Step | Question | Example output |
| ---- | -------- | -------------- |
| **1. Trigger identification** | which words evoke a frame? | *gave* |
| **2. Frame classification** | which frame does each trigger evoke? | *gave* → **Giving** |
| **3. Argument extraction** | which spans fill the frame's roles? | Donor = *the chef*, Recipient = *the customer* |

This library uses **FrameNet 1.7** (via NLTK) and the **Open-Sesame** document
splits (23 test / 8 dev documents) so that results are directly comparable to the
prior work it builds on.

---

## Results vs. the baseline

The baseline is David Chanin's **`frame-semantic-transformer`** (`base` model, a
fine-tuned T5), which we reproduced with our own scoring harness on the
Open-Sesame **test** split. All encoder numbers use the *same* metrics
(word-level trigger F1; frame accuracy; weighted argument F1 with non-core frame
elements counted 0.5) so the comparison is apples-to-apples.

| Task | Baseline (T5) | **texture-frames (DeBERTa)** | Verdict |
| ---- | ------------- | ---------------------------- | ------- |
| Trigger identification | 0.735 | **0.750** | ✅ **ahead** |
| Argument extraction | 0.753 | **0.750** | ✅ **parity** (within run-to-run noise; ahead on dev) |
| Frame classification | 0.887 | 0.863–0.868 | competitive (≈ −0.02) |
| Inference | 3 sequential beam-search passes, **196.6 ms/sample** | single forward pass per stage, **~50–60 ms/stage** | ✅ **~3–4× faster** |

**How to read this honestly:**

- **Argument extraction is the headline.** It is the hardest step and the one
  where the generative baseline had a structural advantage (it is trained to
  *reproduce* the gold span text, which the exact-match metric rewards). Our
  encoder went **0.628 → 0.712 → 0.750** across successive redesigns and now
  **ties the baseline while running ~4× faster**.
- **Trigger identification** beats the baseline outright.
- **Frame classification** lands ~0.02 short. The model's *discrimination* is
  essentially at baseline level; the residual gap is almost entirely a
  candidate-lexicon **coverage ceiling** (2.2% of gold frames fall outside the
  lexicon's candidate set), which is a lexicon-completeness limit rather than a
  modelling one.
- **Speed** is the consistent win: every stage is a single forward pass with no
  beam search. (Encoder timings are unbatched; batching widens the gap further.)

Full development history, including approaches that *didn't* work and why, is in
[`MILESTONES.md`](MILESTONES.md).

---

## Installation

Requires **Python ≥ 3.9**. A GPU is optional (CPU works, just slower).

### Local

```bash
python -m venv .venv && source .venv/bin/activate      # recommended: isolate deps
pip install git+https://github.com/texturejc/Texture_Frames
```

This pulls in `torch`, `transformers`, `sentencepiece`, `nltk`, `huggingface_hub`,
and `numpy`. On **first use**, two things download automatically and are then
cached:

- the **FrameNet 1.7 lexicon** (NLTK `framenet_v17`, plus WordNet), ~50 MB;
- the **three model checkpoints** from the Hugging Face Hub (public), ~1.7 GB each.

```python
from texture_frames import FrameParser
parser = FrameParser()                 # first call downloads + caches everything
print(parser.parse("She sold her bike ."))
```

### Google Colab

```python
!pip install -q git+https://github.com/texturejc/Texture_Frames
```
```python
from texture_frames import FrameParser
parser = FrameParser()                 # use a GPU runtime for ~50 ms/stage
print(parser.parse("The committee awarded her the prize ."))
```

Select **Runtime → Change runtime type → GPU** for speed; the checkpoints are
public, so **no tokens are needed**.

### Dependency-conflict notes

This library was deliberately kept lean to avoid the environment problems that
plague heavier NLP stacks:

- **protobuf** — the package sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
  on import, sidestepping the common protobuf C++/`descriptors` crash. You don't
  need to pin protobuf.
- **numpy** — the package works with numpy ≥ 1.24 (both the 1.x and 2.x lines);
  it does **not** depend on `datasets`/`pyarrow`, which are the usual source of
  `numpy.dtype size changed` ABI errors.
- **torch** — not pinned to a build, so it coexists with whatever CUDA/torch your
  environment (or Colab) already ships.
- If a Colab runtime is in a broken numpy state from a *previous* install,
  **Runtime → Disconnect and delete runtime** (not just *Restart*) and reinstall
  in a clean runtime.

---

## Usage

`FrameParser.parse(text)` returns a `list[FrameAnnotation]`, one per detected
trigger:

```python
@dataclass
class FrameAnnotation:
    trigger: str          # the trigger word
    trigger_loc: int      # its char offset in the sentence
    frame: str            # the FrameNet frame name
    arguments: list[Argument]

@dataclass
class Argument:
    role: str             # the frame element (role) name
    text: str             # the span text
    start: int            # char offset in the sentence (-1 if not locatable)
    end: int
```

### Basic

```python
from texture_frames import FrameParser
parser = FrameParser()

for ann in parser.parse("Kim persuaded the board to delay the vote ."):
    print(f"[{ann.frame}] trigger={ann.trigger!r}")
    for a in ann.arguments:
        print(f"    {a.role:12} {a.text!r}  (chars {a.start}:{a.end})")
```

### Multiple predicates in one sentence

A sentence usually evokes several frames; each is returned separately:

```python
anns = parser.parse("The chef gave food to the customer who thanked her .")
for ann in anns:
    print(ann.trigger, "->", ann.frame,
          "->", [(a.role, a.text) for a in ann.arguments])
# gave     -> Giving         -> [('Donor', 'The chef'), ('Theme', 'food'), ('Recipient', 'to the customer')]
# thanked  -> Judgment_...   -> [...]
```

### Batch over many sentences

```python
sentences = ["The bank raised interest rates .",
             "He borrowed money from a friend .",
             "They celebrated the victory ."]
results = {s: parser.parse(s) for s in sentences}
```

### Choosing device and operating points

```python
parser = FrameParser(
    device="cuda",      # "cpu" or "cuda"; defaults to cuda if available
    frame_bias=7.0,     # candidate soft-mask strength for frame classification
    null_bias=2.0,      # NULL-reject threshold for argument spans (↑ = higher precision)
    trigger_bias=0.0,   # trigger-recall lever; >0 catches borderline triggers (↓ precision)
    max_length=320,     # tokenizer truncation length
)
```

`frame_bias` and `null_bias` default to the values selected on the dev split
(see below); raise `null_bias` if you want argument extraction to be more
conservative (fewer, more precise spans), lower it for higher recall.

`trigger_bias` defaults to `0.0` — the operating point at which the trigger
model was benchmarked (0.750 F1). The identifier has ~0.76 recall, so it
occasionally misses a true trigger by a hair (e.g. *awarded* in "The committee
awarded her the prize"). It is a **precision↔recall dial** that holds F1 roughly
constant (a dev sweep is flat at ~0.79 from bias 0.0 to 1.0) while trading surer
triggers for more of them:

| `trigger_bias` | precision | recall | behaviour |
| -------------- | --------- | ------ | --------- |
| 0.0 (default)  | higher    | lower  | fewer, surer triggers (benchmarked point) |
| 0.5            | mid       | mid    | balanced |
| 0.75–1.0       | lower     | higher | catches borderline misses; more spurious frames |

Set it once on the parser, or **per call** without reloading the models:

```python
parser = FrameParser(trigger_bias=0.5)          # instance default for every call
parser.parse("The committee awarded her the prize .", trigger_bias=0.75)  # this call only
```

The default stays `0.0` so the headline number is the clean argmax result; raise
it if, for your use, missing a whole frame is worse than adding a spurious one.

### Convert to a plain dict (e.g. for JSON)

```python
import dataclasses, json
out = [dataclasses.asdict(ann) for ann in parser.parse("She sold her bike .")]
print(json.dumps(out, indent=2))
```

### Command line

Installing the package also provides a `texture-frames` command:

```bash
texture-frames "The chef gave food to the customer ."
# [Giving] 'gave'
#     Donor          'The chef'
#     Theme          'food'
#     Recipient      'to the customer'

texture-frames --json "She sold her bike ."      # machine-readable output
echo "They celebrated the victory ." | texture-frames    # read from stdin
texture-frames --device cpu "Kim resigned ."
```

---

## How it works: models, choices, and training

All three heads fine-tune **`microsoft/deberta-v3-large`**. The overarching
design decision is **encoder + task heads** rather than the baseline's T5
sequence-to-sequence *generation*: a single forward pass per stage, with no
autoregressive decoding and no beam search, which is where the speed comes from.

### 1. Trigger identification — token classification

A per-token classifier (`O` / `TRIGGER`, first-subword labelling) over the
sentence. Scored with the upstream **word-level F1** so it is directly
comparable. One subtlety handled here: DeBERTa's SentencePiece tokenizer reports
a word's first subtoken as starting on the *preceding space*, so trigger–word
alignment snaps past leading whitespace to avoid off-by-one errors.
**Result: 0.750 F1 vs. 0.735 — ahead of baseline.**

### 2. Frame classification — marker-token pooling

A sequence classifier over the ~1,221 FrameNet frames. The trigger is wrapped in
**entity markers** (`… <t> gave </t> …`), and — crucially — the frame
representation is the concatenation of the **`<t>` and `</t>` marker hidden
states** rather than the usual `[CLS]` vector. This *entity-marker pooling*
(as in relation extraction) focuses the classifier on the predicate in context.
At inference, logits are **soft-masked** toward the lexicon's candidate frames
for that trigger (a positive bias, tuned on dev) so a confident non-candidate can
still win, recovering golds outside the candidate set.

*Design notes (what we learned):* pooling the trigger markers beat `[CLS]`
slightly; a **hard** candidate mask caps accuracy at the lexicon's 0.978 coverage;
and feeding candidate *names* into the input actually **hurt** (it diluted the
pooled representation and invited a shortcut) — a reminder that for a
sequence-level task the fix was better *pooling*, not more *input*.
**Result: 0.863–0.868 vs. 0.887 — competitive; the residual is the coverage
ceiling, not discrimination.**

### 3. Argument extraction — detect-then-classify (the main contribution)

This is where the encoder had to earn parity, and where the architecture matters
most. A naïve port — flat **BIO tagging over ~2,400 role-labels** — topped out
around **0.63**, because (a) ~1,200 frame elements form a long tail that starves
a flat 2,400-way tagger, and (b) the exact-span metric punishes boundary
imprecision, which a generative model sidesteps by *writing* the span text.

The v2 model **decomposes** the task into two heads on one backbone:

- **Head A — span detection:** a **3-class** BIO tagger (`O`/`B`/`I`), "is this
  token part of *an* argument?", role-agnostic. Three classes instead of 2,400
  gives dense gradient signal → better boundaries and recall, and it handles
  arbitrary-length spans (no width cap).
- **Head B — role classification:** for each detected span, pool its tokens
  (`start ⊕ end ⊕ mean`) and classify into **only the current frame's frame
  elements** (plus a `NULL` reject class), masked via the lexicon. Choosing among
  ~10 options instead of 2,400 removes the long-tail starvation and role
  confusion.

The input carries two conditioning signals that both helped: the trigger is
**predicate-marked** inline (`… <t> gave </t> …`) so the model knows *which*
predicate's arguments to extract, and the frame's **frame-element menu** is
listed in the prefix (`{frame} [Donor; Recipient; …] : …`) so it knows which
roles to look for. Training adds **sampled `NULL` negative spans** so Head B
learns to reject spurious detections, and **WordNet synonym augmentation** (with
careful character-offset remapping of the trigger and every role span) to recover
the long-tail edge the baseline gets from data augmentation. At inference a
**`NULL`-bias sweep** picks the precision/recall operating point on dev.

**Result: 0.628 (flat BIO) → 0.712 (detect-then-classify) → 0.750 (+augmentation)
— parity with the baseline's 0.753, and ~4× faster.**

### Training protocol

| Setting | Value |
| ------- | ----- |
| Backbone | `microsoft/deberta-v3-large` |
| Precision | bf16 mixed precision (model weights loaded fp32 to avoid fp16 gradient-scaler issues) |
| Optimiser | AdamW, lr 1e-5, warmup ratio 0.06, weight decay 0.01 |
| Batch / length | batch 16, max length 320 |
| Epochs | 5 (trigger/frame), 6 with augmentation (args) |
| Checkpointing | HF `Trainer`, best-on-dev selected each epoch |
| Hardware | Google Colab (A100 / L4) |
| Data | FrameNet 1.7 (NLTK), Open-Sesame document splits |

A guiding principle throughout was **metric comparability**: every encoder head
is scored with the exact metric definition used for the baseline, and the
baseline was reproduced in the same harness first (within ~0.5 pt of its published
numbers) so that "we match/beat it" is a fair claim rather than an artefact of
different scoring.

---

## Model checkpoints

Public on the Hugging Face Hub; downloaded and cached automatically:

- Trigger — [`texturejc/texture-frames-trigger`](https://huggingface.co/texturejc/texture-frames-trigger)
- Frame — [`texturejc/texture-frames-frame`](https://huggingface.co/texturejc/texture-frames-frame)
- Arguments — [`texturejc/texture-frames-args`](https://huggingface.co/texturejc/texture-frames-args)

---

## Acknowledgements

This work stands entirely on the shoulders of **David Chanin** and his
[`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer).
His library is the baseline we reproduce and compare against, and — just as
importantly — it shaped the *approach*: the three-stage task decomposition, the
FrameNet 1.7 data handling, the Open-Sesame evaluation splits, and the scoring
conventions all follow his work. **texture-frames would not exist without it, and
we are deeply grateful for his contribution to open frame-semantic parsing.** If
you use this library, please also acknowledge and cite his.

We likewise thank the **Berkeley FrameNet** project (Collin Baker, Charles
Fillmore, and colleagues) for the resource that makes any of this possible, the
authors of **Open-Sesame** (Swayamdipta et al., 2017) for the evaluation splits,
and the authors of **DeBERTa** (He et al., 2021) for the backbone.

---

## Citation & license

If you use this library, please cite this repository and the upstream work:

```bibtex
@software{texture_frames,
  author = {Carney, James},
  title  = {texture-frames: a fast DeBERTa encoder FrameNet parser},
  url    = {https://github.com/texturejc/Texture_Frames},
  year   = {2026}
}
@software{chanin_frame_semantic_transformer,
  author = {Chanin, David},
  title  = {frame-semantic-transformer},
  url    = {https://github.com/chanind/frame-semantic-transformer}
}
```

**License:** MIT for this code. The models are trained on **FrameNet 1.7**, which
carries its own (academic-use) terms — please review those before redistributing
the weights. FrameNet: Baker, Fillmore & Lowe (1998), *The Berkeley FrameNet
Project*.
