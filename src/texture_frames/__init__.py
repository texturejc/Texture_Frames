"""texture-frames — fast DeBERTa-based FrameNet semantic frame parser.

    from texture_frames import FrameParser
    parser = FrameParser()                       # downloads weights on first use
    for ann in parser.parse("The chef gave food to the customer ."):
        print(ann.frame, ann.trigger, ann.arguments)
"""
from .pipeline import Argument, FrameAnnotation, FrameParser

__all__ = ["FrameParser", "FrameAnnotation", "Argument"]
__version__ = "0.1.1"
