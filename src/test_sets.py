"""Named evaluation sets for the Test runs page.

SMOKE_SET    — 5 calls, one per call type, for a fast sanity check.
REGRESSION_SET — 9 calls, covering high- and low-quality examples across
                 all call types, for a fuller rubric regression.
"""

SMOKE_SET: frozenset[str] = frozenset({
    "HB-2026-00203",   # insurance_verification — low score, seed
    "HB-2026-00512",   # billing_inquiry — high score, seed
    "HB-SYNTH-00120",  # prescription_refill — mid score, synthetic
    "HB-2026-00401",   # clinical_triage — low score, only available
    "HB-2026-00315",   # prescription_refill — high score, seed
})

REGRESSION_SET: frozenset[str] = frozenset({
    "HB-2026-00147",   # appointment_scheduling — high, seed
    "HB-2026-00289",   # billing_inquiry — high, seed
    "HB-SYNTH-00118",  # billing_inquiry — high, synthetic
    "HB-SYNTH-00119",  # billing_inquiry — low, synthetic
    "HB-2026-00401",   # clinical_triage — only available
    "HB-2026-00203",   # insurance_verification — low, seed
    "HB-2026-00315",   # prescription_refill — high, seed
    "HB-2026-00587",   # prescription_refill — low, seed
    "HB-SYNTH-00120",  # prescription_refill — mid, synthetic
})
