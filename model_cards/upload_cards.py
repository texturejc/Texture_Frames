"""
Push the three model cards to their Hugging Face repos, and (optionally) delete
the leftover training-checkpoint folders that bloat each repo by ~5 GB.

Run after authenticating a *write* token:

    huggingface-cli login
    python model_cards/upload_cards.py            # push cards only
    python model_cards/upload_cards.py --cleanup  # also delete checkpoint-*/ dirs
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, list_repo_files, upload_file

HERE = os.path.dirname(os.path.abspath(__file__))

CARDS = {
    "texturejc/texture-frames-trigger": "trigger/README.md",
    "texturejc/texture-frames-frame": "frame/README.md",
    "texturejc/texture-frames-args": "args/README.md",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleanup", action="store_true",
                    help="also delete leftover checkpoint-*/ training-state folders")
    args = ap.parse_args()
    api = HfApi()

    for repo, card in CARDS.items():
        upload_file(
            path_or_fileobj=os.path.join(HERE, card),
            path_in_repo="README.md", repo_id=repo, repo_type="model",
        )
        print("card ->", repo)

    if args.cleanup:
        for repo in CARDS:
            ckpt_dirs = sorted({
                f.split("/")[0] for f in list_repo_files(repo)
                if f.startswith("checkpoint-")
            })
            for d in ckpt_dirs:
                api.delete_folder(path_in_repo=d, repo_id=repo, repo_type="model")
                print("deleted", d, "from", repo)


if __name__ == "__main__":
    main()
