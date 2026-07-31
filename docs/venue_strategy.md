# Post-AAAI venue strategy

_Snapshot: 2026-07-31, the day AAAI-27 supplementary closed._

Companion to `docs/future.md` (research roadmap). This document is about
**sequencing**: what we are legally allowed to submit where, when, and what to
build in the meantime.

## 1. The binding constraint: the AAAI-27 review window

The paper went in under the AAAI-27 main technical track (abstracts 2026-07-21,
full papers 2026-07-28, supplementary/code 2026-07-31). The AAAI-27 dual
submission policy states that AAAI does not permit simultaneous submission of
work with an overlapping author set that does not constitute a *distinct*
scientific contribution, to AAAI-27 or to any other archival conference or
journal. It further warns that if publishing one paper would render another
insufficiently novel, the AAAI submission may be summarily rejected.

**Consequence: this manuscript is frozen for archival resubmission until the
AAAI decision is final.** The relevant AAAI-27 dates:

| Milestone | Date |
|---|---|
| Phase-1 (early) rejection notification | 2026-09-24 |
| Author response period | 2026-10-19 – 2026-10-25 |
| Final accept/reject notification | 2026-11-30 |

Two explicit escape valves the policy *does* allow:

- **arXiv and other preprint servers are permitted.**
- **Non-archival workshops are permitted.**

## 2. Venue calendar, filtered by that constraint

| Venue | Deadline | Verdict |
|---|---|---|
| WACV 2027 Round 2 | 2026-08-28 | **Blocked.** Inside the AAAI window; a dual-submission violation. |
| ISBI 2027 (4 pp) | 2026-10-26 | **Conditional.** Legal only if AAAI phase-1 rejects on 09-24. |
| CVPR 2027 | ~2026-11 | **Risky.** Almost certainly lands before the 11-30 final notification. |
| **IPMI 2027** (12 pp LNCS) | **2026-12-07** | **Primary target.** First clean date after 11-30. |
| MIDL 2027 | ~2026-12 / 2027-01 | Clean; confirm dates when posted. |
| **MICCAI 2027** (8 pp) | **~2027-02-26** | **Backup.** Clean, large audience. |

### Why not WACV

Beyond the legal problem, WACV is the wrong shape for this paper. The two
load-bearing contributions are (a) a provable statement about estimators and
(b) a task-based clinical detectability evaluation. A CV applications track
undersells both. The `docs/future.md` "Option B / WACV applications paper"
framing predates the theorem and should be considered superseded.

### Why IPMI first

- First deadline that is unambiguously clear of the AAAI lock.
- 12 pages excluding references — room for the full theory *plus* every
  robustness arm, with nothing exiled to a technical appendix.
- IPMI's identity is methodological and theoretical rigor in medical imaging,
  which is exactly this paper's shape.
- Biennial, so the prestige is high and the next slot is 2029.

**The risk is the 7-day gap** between the AAAI final notification (11-30) and
the IPMI deadline (12-07). This is only survivable if the extended manuscript
is written during Sept–Nov *regardless of the AAAI outcome*. Treat the
extension as unconditional work, not as contingency work.

## 3. What to add

Ranked by (value to the paper) / (cost). Tier 1 items do **double duty**: they
are also the experiments reviewers are most likely to demand, so having them in
hand before the 2026-10-19 author-response window directly improves the AAAI
odds too.

### Tier 1

**(1) Sinogram-domain intervention.** The single highest-value experiment.
The paper currently claims that FBP-correlated noise is *why* blind-spot
self-supervision fails on CT — but that claim is correlational, supported by
the natural-image i.i.d. contrast. Running the identical N2V / Noise2Sim /
SSFlow comparison in the *projection* domain, where per-detector noise is
genuinely pixel-independent, converts it into a controlled intervention: N2V
should recover, and the exclusion radius `r` should stop mattering exactly as
it does on i.i.d. natural images. The TCIA
`LDCT-and-Projection-data` collection we already pull ships the projection
data, so this needs a projection loader and an FBP, not a new dataset.
_Cost: medium-high. Value: very high._

**(2) Generalize the finite-step departure beyond the Gaussian model.** The
limitations section already concedes this: the departure proposition is exact
only in the jointly-Gaussian per-mode model. A reviewer who wants to call the
theory thin will aim here. Even a partial result lands — monotonicity under a
log-concave prior, or a risk-increase lower bound in terms of the score's
Lipschitz constant, or a Tweedie-based treatment of the non-Gaussian case.
This is analytical work with zero compute cost and it is what would make the
theory section stand on its own at IPMI.
_Cost: analysis only. Value: very high._

**(3) A second, independent scanner/cohort.** Everything is currently one TCIA
collection — abdomen (50 patients) and chest (10 patients) from the same
acquisition family. The limitations section names multi-scanner replication as
the natural next step, so reviewers will too. The Mayo/AAPM-2016 LDCT
challenge data and LoDoPaB-CT are the obvious candidates.
_Cost: medium (acquisition + one sweep). Value: high — retires the most
predictable objection._

### Tier 2 — cheap, reuses the existing CHO harness

**(4) A second observer model.** The detectability erosion (7–8% at 160 HU) is
the paper's most contestable empirical claim and it rests on a single observer.
Adding NPWE and/or a CHO with internal noise, plus a proper significance test,
makes it robust for very little work.

**(5) Denser lesion grid.** More contrasts and 2–3 lesion radii, to map where
erosion begins rather than reporting three points. Same harness, more compute.

**(6) Close the label-free gap.** The weakest headline number is that our best
label-free result (+1.99 dB) beats the plain Noise2Sim baseline (+1.83 dB) by
0.16 dB. That reads as a tie, and reviewers will say so. Anything that moves
this — stronger backbone, longer schedule, a PMRF-style two-stage readout,
ensembling pairings across `r` — changes the story from "we match the baseline"
to "we set the label-free bar."

### Tier 3 — only with time and budget

**(7) Reader study.** Per `docs/future.md`, a multi-reader confirmation would
escalate the detectability finding from methodological caveat to patient-safety
result. Expensive; needs radiologists.

**(8) Cross-dose / cross-kernel generalization, head anatomy.** Already tracked
in `todo.md`.

## 4. Recommended timeline

| Window | Work |
|---|---|
| Aug – Sep 2026 | Tier 1: sinogram arm, theory generalization, second cohort. This is rebuttal ammunition. |
| 2026-09-24 | AAAI phase-1 notification. If rejected, ISBI (10-26) opens as a fast 4-page option. |
| 2026-10-19 – 10-25 | AAAI author response. Cite the new results. |
| Oct – Nov 2026 | Write the extended 12-page IPMI manuscript **unconditionally**. |
| 2026-11-30 | AAAI final notification. |
| 2026-12-07 | IPMI submission if AAAI rejected. |
| 2027-02-26 | MICCAI fallback (compress the IPMI manuscript to 8 pages). |

## 5. Things worth doing now, at zero legal risk

- **Release the repository.** The paper promises it ("repository link withheld
  for anonymous review"); publish once anonymity no longer binds.
- **Consider an arXiv preprint.** Explicitly permitted by AAAI-27. Check the
  publicity/anonymity rules before posting, since permission to post is not
  necessarily permission to promote during review.
- **Consider a non-archival workshop.** Also explicitly permitted, and it gets
  the work in front of the medical-imaging community during the lock without
  burning archival novelty.
- **Do not split the paper.** Salami-slicing the detectability section into a
  separate submission during the window is exactly what the "insufficiently
  novel" clause is aimed at. Keep one strong manuscript.
