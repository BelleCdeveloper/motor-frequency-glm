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

## Results
- Test Poisson deviance (exposure-weighted): 0.598, against 0.627 for an intercept-only baseline, a reduction of about 4.7%.
- Actual vs expected claims on the test set: 7,239 vs 7,172 (expected about 0.9% below actual, within random variation).
- Bonus-malus is the dominant factor. Relative to bonus-malus 50, expected frequency is about 1.3x at 51-60, 2.0x at 61-75, 2.3x at 76-100, 5.4x at 101-125 and 6.8x at 126+.
- Vehicle age lowers frequency (about 0.66x for 16+ years vs 6-10 years).
- Driver age effects are modest and conditional on bonus-malus, since the two are correlated.

## Limitations
- Overdispersion: Pearson dispersion of 2.27 on the training data means Poisson standard errors are understated by roughly 1.5x.
- Single random train/test split, not a time-based or cross-validated one.
- No interaction terms (for example between driver age and bonus-malus), and bands were chosen by hand.
- Frequency only: no severity model, so no pure premium yet.
- Public data is cleaner than a real portfolio.

## Next steps
Severity GLM and pure premium, a LightGBM comparison, and a simple pricing API.
