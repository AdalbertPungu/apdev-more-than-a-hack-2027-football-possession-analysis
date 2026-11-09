# Technical note - Phase 1 possession solution

## Approach

This deterministic pipeline uses tactical Automatic Tracking Data (ATD) and automatic Smart
Tagging to infer possession and stoppages. It separates three decisions: whether control is
supported, whether play has stopped, and whether a restart is observed. Insufficient or stale
evidence produces abstention instead of an indefinitely retained answer.

It is an offline algorithm: sustained runs are confirmed using later frames and backdated to
the start of the supporting run. This avoids adding the confirmation delay to every boundary.
No prediction is manually corrected or compared with the video.

## Assets and training data

Inputs are the tactical ATD metadata, raw tracking, and Smart Tagging XML. ATD frame IDs start
at one, so submission frame = ATD frame ID - 1. Tag times are seconds on the tactical timeline:
frame = round(seconds * 25). Distances use a 105.3 by 68 metre pitch.

Player slots are tracklets, not reliable player identities. The algorithm does not assume that
a slot follows one person across the match. No video, broadcast data, or homography is consumed.
No external training data, human annotations, fitted model, or automatic calibration is used.
No ground truth is read by the generator.

## Controlled possession

Nearest-team distances are measured in metres. A control candidate requires:

- a ball observation with usable speed and no implausible adjacent-frame displacement;
- the nearest player within 3 metres, at least 0.75 metres closer than the opposing team;
- ball speed at most 6 m/s and speed magnitude within 4 m/s of the nearest player's speed;
- signed tracking plus offensive-tag evidence of at least 0.35.

The tracking evidence retains the starter's distance-based principle, with weaker support for
distant players and fast balls. Offensive phases add 0.35 per tag, clipped to a magnitude of
0.7. Defensive phases do not vote for their tagged team. Tags cannot create control without
the geometry and speed checks. Contested proximity, fast passing balls, and absent player
speeds therefore do not independently establish a new possessor.

A candidate must persist for eight frames before acquisition or a team change. Brief opponent
touches and loose-ball intervals retain the established team for up to 125 frames without renewed
control support, then produce `X`. Confirmation is backdated to the candidate run's first frame.
Speed magnitude agreement is a proxy for control, not relative velocity or proof of a touch.

## Stoppages and restarts

A stoppage can be supported by:

1. Ball coordinates more than 0.03 outside the normalised pitch for at least eight frames.
2. At least 25 consecutive frames with at most four tracked players and no tracked ball.
3. At least 25 stationary-ball frames, speed at most 0.5 m/s and positional span at most
   0.5 metres, overlapping an automatic set-piece interval.

A stationary ball alone does not imply dead play. Set-piece tags alone do not mark an entire
interval dead. A single out-of-pitch detection is not enough to stop play.

Once stopped, the decoder requires three consecutive reliable moving-ball frames, speed greater
than 1 m/s and coordinates inside the tolerated pitch bounds, to infer a restart. Its team comes
from a current control candidate, an unambiguous set-piece team, or a control candidate observed
within the previous 50 frames while stopped. Conflicting restart tags and `N/A` provide no team.
Without a known team, confirmed motion leads to `X` rather than inventing ownership.

Missing-ball frames cannot confirm a restart. A stopped state without renewed stoppage support
produces `X` after 125 frames, while internally still requiring restart evidence before resuming.
No centered smoothing blends possession evidence across stoppages.

## Fixed assumptions and compliance

All thresholds are explicit design assumptions, not parameters tuned on this match, its video,
or its ground truth. The ball-speed plausibility ceiling is 45 m/s; adjacent observations above
that implied speed cannot support control or restart. Missing speed is estimated only from two
adjacent observed positions, never across an occlusion.

Behavioral changes are checked with synthetic fixtures rather than selected by public scores.
The earlier public-window control score belongs to the earlier implementation and must not be
reported as this revision's score. No public-ground-truth comparison is used to tune or select
this revision.

## Limitations

The algorithm does not observe referee decisions, actual ball contact, or ball height. Projected
coordinates and speed errors can create false cues. Legitimate short possessions may be missed,
and slow or occluded restarts may be delayed. The absence of stoppage cues is not proof of live
play. A recent contact or automatic restart tag may assign the wrong restart team.

Abstention acknowledges uncertainty but does not erase it from ranking: the official temporal
metric fills predicted `X` gaps, potentially carrying an old label across a real transition.
No improvement in full-match score is claimed. Further validation or parameter tuning would
require permitted external match data.

## Reproduction

Use Python 3.11 or newer with pinned NumPy 2.3.4. Place the three tactical inputs under
`data\tactical`, or supply their root with `--data-dir` or `MTAH_DATA_DIR`.

```powershell
python -m pip install -r requirements.txt
python -W error -m unittest discover -s tests -v
python possession_solution.py --out ApDev.csv
python validate_submission.py ApDev.csv
```

The CSV contains 199,319 rows indexed 0 to 199318. Match data and submissions are not included
in the public source repository. Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
