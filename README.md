# Motor claim frequency: Poisson GLM

A baseline claim frequency model for French motor third-party liability insurance (freMTPL2freq, about 678,000 policies).

## Problem
Estimate the expected number of claims per policy-year from rating factors. This is the first half of a pure premium model (frequency x severity).

## Data
freMTPL2freq, downloaded from OpenML (`data_id=41214`) when the script runs. The data is not stored in this repo.

## Method
- Target: `ClaimNb` (capped at 4), Poisson family with log link
- Exposure: `log(Exposure)` as an offset, with exposure capped at one year
- Rating factors: area, vehicle power, vehicle age, driver age, bonus-malus (continuous variables banded)
- Base level of each factor: the level with the most exposure
- 80/20 random train/test split
- Evaluation: exposure-weighted Poisson deviance, actual vs expected claims, and a dispersion check

## Results
_Fill in after running: test deviance, actual vs expected claims, dispersion, and the key relativities (for example, the effect of bonus-malus and driver age)._

## Limitations
_Fill in: random rather than time-based split, no interactions, banding choices made by hand, frequency only (no severity), and public data that is cleaner than a real portfolio._

## Run it
```bash
pip install -r requirements.txt
python frequency_glm.py
```

## Next steps
Severity GLM and pure premium, a LightGBM comparison, and a simple pricing API.
