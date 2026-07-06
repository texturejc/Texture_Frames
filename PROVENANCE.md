# Provenance

The `frame-semantic-transformer/` directory was vendored from:

- **Repo:** https://github.com/texturejc/frame-semantic-transformer
  (a fork of https://github.com/chanind/frame-semantic-transformer)
- **Commit:** `18cb3023bbb6df0c1b53a52182135c0c0132c073`
- **Tag/version:** `0.10.0`
- **Vendored on:** 2026-07-06

Its original git history has been removed so the outer project repo can track
and checkpoint changes to it directly. To compare against or pull updates from
upstream, add it as a remote against the commit above.

This project (Path B) re-architects the T5-based parser into an encoder + task
heads model (DeBERTa-v3-large) while reusing the original data loaders, the
FrameNet lexicon candidate lookup, the Open-Sesame data splits, and the
evaluation harness. See `MILESTONES.md` for the plan.
