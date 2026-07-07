"""Structural tests for the installable texture_frames package (no weights/nltk)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import texture_frames as tf  # noqa: E402
from texture_frames import Argument, FrameAnnotation, FrameParser  # noqa: E402


def test_exports():
    assert tf.__all__ == ["FrameParser", "FrameAnnotation", "Argument"]
    assert isinstance(tf.__version__, str)


def test_dataclasses():
    a = Argument(role="Donor", text="the chef", start=0, end=8)
    ann = FrameAnnotation(trigger="gave", trigger_loc=9, frame="Giving", arguments=[a])
    assert ann.frame == "Giving"
    assert ann.arguments[0].role == "Donor"
    assert FrameAnnotation(trigger="x", trigger_loc=0, frame="F").arguments == []


def test_parser_has_pipeline_api():
    for m in ("parse", "_triggers", "_frame", "_args", "_allowed_role_ids"):
        assert hasattr(FrameParser, m)


def test_submodules_import():
    for m in ["pipeline", "weights", "cli", "model_args2", "model_frame2", "lexicon",
              "data", "args_data", "args2_data", "frame2_data"]:
        importlib.import_module(f"texture_frames.{m}")


def test_cli_help_exits_zero():
    from texture_frames import cli

    try:
        cli.main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    else:
        raise AssertionError("--help should exit")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\ndone")
