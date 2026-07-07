# texture-frames

A fast FrameNet semantic frame parser: a **DeBERTa-v3-large encoder** rearchitecture
of the T5-based [`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer).
One forward pass per stage instead of three beam-search generation passes —
**~4× faster** — while matching or beating the baseline on trigger identification
and argument extraction (see `MILESTONES.md`).

## Install

```bash
pip install git+https://github.com/texturejc/Texture_Frames
```

Pulls in `torch`, `transformers`, `nltk`, `sentencepiece`, `huggingface_hub`. The
FrameNet 1.7 lexicon (nltk `framenet_v17`) downloads automatically on first use.

## Use

```python
from texture_frames import FrameParser

parser = FrameParser()          # downloads the 3 model checkpoints on first use (cached)
for ann in parser.parse("The chef gave food to the customer ."):
    print(ann.frame, "|", ann.trigger)
    for arg in ann.arguments:
        print("   ", arg.role, "->", arg.text)
```

```
Giving | gave
    Donor -> The chef
    Theme -> food
    Recipient -> to the customer
```

`parse()` returns a list of `FrameAnnotation(trigger, trigger_loc, frame, arguments)`,
each `arguments` a list of `Argument(role, text, start, end)`.

### Options

```python
FrameParser(
    device="cuda",           # default: cuda if available else cpu
    frame_bias=7.0,          # candidate soft-mask bias (dev-selected)
    null_bias=2.0,           # args NULL-reject threshold (dev-selected)
)
```

## Model weights

The three checkpoints live on the Hugging Face Hub and download on first use:
`texturejc/texture-frames-{trigger,frame,args}`. If those repos are **private**,
authenticate once (`huggingface-cli login` or set `HF_TOKEN`); make them public
for auth-free installs.

### Publishing the weights (run once, from Colab after training)

```python
from huggingface_hub import login, create_repo, upload_folder
login()  # paste an HF *write* token

for name, local in [
    ("trigger", "/content/outputs/trigger"),   # AutoModelForTokenClassification (save_pretrained)
    ("frame",   "/content/outputs/frame2"),     # frame2_model.pt + tokenizer + frame2id.json
    ("args",    "/content/outputs/args2"),       # args2_model.pt   + tokenizer + role2id.json
]:
    repo = f"texturejc/texture-frames-{name}"
    create_repo(repo, private=True, exist_ok=True)
    upload_folder(folder_path=local, repo_id=repo)
    print("uploaded", repo)
```

The repo names must match `DEFAULT_*_REPO` in `texture_frames/pipeline.py`.

## Layout

- `src/texture_frames/` — the installable package (inference only)
- `encoder_parser/` — training/eval code + Colab notebooks (not shipped)
- `MILESTONES.md` — full development record and benchmark results

## License

MIT for this code. Note the models are trained on FrameNet 1.7 (its own
academic-use terms) — confirm those before public redistribution of the weights.
