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
- Baseline: an intercept-only Poisson model (one average frequency for everyone) for comparison
- Standard errors corrected for overdispersion using a quasi-Poisson scale (Pearson chi-square / df = 2.27)

## Results
- Test Poisson deviance (exposure-weighted): 0.598, against 0.627 for the intercept-only baseline, a reduction of about 4.7%.
- Actual vs expected claims on the test set: 7,239 vs 7,172 (expected about 0.9% below actual, within random variation).
- Bonus-malus is the dominant factor. Relative to bonus-malus 50, expected frequency is about 1.3x at 51-60, 2.0x at 61-75, 2.3x at 76-100, 5.4x at 101-125 and 6.8x at 126+.
- Vehicle age lowers frequency (about 0.66x for 16+ years vs 6-10 years).
- Driver age effects are modest and conditional on bonus-malus, since the two are correlated.
- After the overdispersion correction, bonus-malus and most vehicle-age effects remain highly significant. Driver age 71+ (p = 0.26) and vehicle age 3-5 (p = 0.22) are not distinguishable from their base levels, and driver age 21 and under is borderline (p = 0.07).
- Open question: vehicle age 0 has a relativity of about 3.2 vs the 6-10 year base, which looks high. This needs checking against exposure and claim counts for that band.

## Limitations
- Overdispersion (Pearson dispersion 2.27) is handled with quasi-Poisson standard errors, which is not a full probability model, so there is no likelihood or AIC. A negative binomial model would be the next thing to test.
- Single random train/test split, not a time-based or cross-validated one.
- No interaction terms (for example between driver age and bonus-malus), and bands were chosen by hand.
- Frequency only: no severity model, so no pure premium yet.
- Public data is cleaner than a real portfolio.

## Run it
```bash
pip install -r requirements.txt
python frequency_glm.py
```

## Changelog
- Initial Poisson GLM with exposure offset and train/test evaluation
- Added intercept-only baseline model and deviance comparison
- Added quasi-Poisson standard errors to correct for overdispersion
- Documented results and limitations

## Next steps
Investigate the vehicle age 0 result, add a severity GLM and pure premium, compare against a LightGBM model, and build a simple pricing API.
