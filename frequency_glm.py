import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split

# 1. Load the French motor frequency data
df = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
df["Area"] = df["Area"].astype(str).str.strip("'\"")
for c in ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus"]:
    df[c] = pd.to_numeric(df[c])

# 2. Clean and band the rating factors
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["VehPower"] = df["VehPower"].clip(upper=9).astype(int).astype(str)
df["VehAge"] = pd.cut(df["VehAge"], [-1, 0, 2, 5, 10, 15, 200],
                      labels=["0", "1-2", "3-5", "6-10", "11-15", "16+"]).astype(str)
df["DrivAge"] = pd.cut(df["DrivAge"], [0, 21, 25, 30, 40, 50, 60, 70, 200],
                       labels=["<=21", "22-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]).astype(str)
df["BonusMalus"] = pd.cut(df["BonusMalus"], [0, 50, 60, 75, 100, 125, 1000],
                          labels=["50", "51-60", "61-75", "76-100", "101-125", "126+"]).astype(str)

# 3. Split, then fit the Poisson GLM with a log-exposure offset
train, test = train_test_split(df, test_size=0.2, random_state=42)
factors = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus"]
terms = []
for f in factors:  # base level = level with most exposure
    base = train.groupby(f)["Exposure"].sum().idxmax()
    terms.append(f'C({f}, Treatment(reference="{base}"))')
model = smf.glm("ClaimNb ~ " + " + ".join(terms), data=train,
                family=sm.families.Poisson(), offset=np.log(train["Exposure"])).fit()
print(model.summary())

# 4. Evaluate on the test set
test = test.copy()
test["Expected"] = model.predict(test, offset=np.log(test["Exposure"]))
print("Test deviance:", mean_poisson_deviance(test["ClaimNb"] / test["Exposure"],
      test["Expected"] / test["Exposure"], sample_weight=test["Exposure"]))
print("Actual vs expected claims:", test["ClaimNb"].sum(), round(test["Expected"].sum()))
print("Dispersion:", model.pearson_chi2 / model.df_resid)

# 5. Relativities (what pricing teams actually read)
print(np.exp(model.params).drop("Intercept").round(3))



null = smf.glm("ClaimNb ~ 1", data=train, family=sm.families.Poisson(),
               offset=np.log(train["Exposure"])).fit()
mu0 = null.predict(test, offset=np.log(test["Exposure"]))
print("Baseline deviance:", mean_poisson_deviance(
    test["ClaimNb"] / test["Exposure"], mu0 / test["Exposure"],
    sample_weight=test["Exposure"]))


