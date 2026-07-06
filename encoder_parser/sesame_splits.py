"""Open-Sesame FrameNet 1.7 document splits.

Vendored verbatim from the upstream parser
(`frame_semantic_transformer/data/loaders/framenet17/sesame_data_splits.py`,
originally from https://github.com/swabhs/open-sesame) so the encoder pipeline
can select the identical train/dev/test documents WITHOUT importing anything
under the `framenet17` package — whose __init__ pulls in the augmentation
modules and therefore `nlpaug`. These lists are static test-set definitions and
do not change across FrameNet 1.7.
"""

SESAME_TEST_FILES = [
    "ANC__110CYL067.xml",
    "ANC__110CYL069.xml",
    "ANC__112C-L013.xml",
    "ANC__IntroHongKong.xml",
    "ANC__StephanopoulosCrimes.xml",
    "ANC__WhereToHongKong.xml",
    "KBEval__atm.xml",
    "KBEval__Brandeis.xml",
    "KBEval__cycorp.xml",
    "KBEval__parc.xml",
    "KBEval__Stanford.xml",
    "KBEval__utd-icsi.xml",
    "LUCorpus-v0.3__20000410_nyt-NEW.xml",
    "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
    "LUCorpus-v0.3__enron-thread-159550.xml",
    "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
    "LUCorpus-v0.3__SNO-525.xml",
    "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
    "Miscellaneous__Hound-Ch14.xml",
    "Miscellaneous__SadatAssassination.xml",
    "NTI__NorthKorea_Introduction.xml",
    "NTI__Syria_NuclearOverview.xml",
    "PropBank__AetnaLifeAndCasualty.xml",
]

SESAME_DEV_FILES = [
    "ANC__110CYL072.xml",
    "KBEval__MIT.xml",
    "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
    "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
    "Miscellaneous__Hijack.xml",
    "NTI__NorthKorea_NuclearOverview.xml",
    "NTI__WMDNews_062606.xml",
    "PropBank__TicketSplitting.xml",
]
