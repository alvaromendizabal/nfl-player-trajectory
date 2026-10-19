# Data card

## Which data this project actually uses

This is the [NFL Big Data Bowl 2026 Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction).
Its verified download inventory contains **49 files**: 18 weekly input/label pairs
for the **2023 NFL season**, 11 organizer API files, and two unlabelled sample files.
The inventory SHA-256 is
`938be8c1cc31ded95e82cf473fbbd7623931ba579adf76732056ff8dabf77b0c`.
The completed audit records **4,880,579 observed player-frames**, **562,936 forecast
player-frames**, and **272 games**. Each raw file has its own checksum.

The [NFL event overview](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
mentions 2023 and 2024 data. That broad event description does not establish
additional labelled Prediction training files. The downloaded Prediction inventory
contains 2023 labels; later calendar years occur in its unlabelled gateway sample.
We do not substitute Analytics-track labels or a third-party data mirror.
January games retain the season encoded in the organizer's filename: a January
2024 date does not create a second labelled season.

## Prediction-time information

The organizer supplies pre-throw tracking, requested future frame IDs, forecast
horizon, player role, and the ball landing point. Each target is the player's
future x/y location in yards. The ball landing point is legitimate task input;
this project does not claim to infer an unknown landing point in a live system.

Coordinates are aligned consistently with play direction. Position-derived motion
uses observed frame intervals at 10 Hz. Reported speed, acceleration, direction,
orientation, and body/position metadata are optional feature inputs with explicit
dependency contracts and independently fitted availability profiles. Identifiers
join frames and build strictly earlier-date histories; they are not arbitrary
numeric predictors. Future player x/y coordinates never enter feature construction.

## Validation and limits

The final neural evaluation uses game-grouped out-of-fold predictions across 272 games. Its supported and complete populations are reported separately in [Results](RESULTS.md). The public synthetic demo uses invented games and is not a measurement on this competition data.

Only one labeled season is established by the audited inventory. Calendar-year
diversity in an unlabelled gateway sample checks interface coverage, not accuracy
on a second season. Repeatedly inspected OOF cannot be reused as an untouched test.

## Access and reproducibility

Competition data has its own terms; the repository's MIT license covers code.
Raw tracking, private frame-error files, credentials, and fitted artifacts are
excluded from the new closeout publication. A small fitted numerical reference is public; the private neural champion weights remain withheld. Public outputs contain aggregate metrics, feature definitions,
and review figures. Content-addressed private snapshots and source/input receipts
support checksum-verified recovery without publishing the competition data.
