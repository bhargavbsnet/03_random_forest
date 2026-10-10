"""
Model 3: RANDOM FOREST
----------------------
Builds hundreds of decision trees, each on a slightly different random sample of
the data, and averages their predictions.

A decision tree asks yes/no questions ("Is it before 13:00? Is it Sunday?"), so
it naturally learns combinations that linear regression cannot. Averaging many
trees stops any single tree from memorising the training data.

Run:  python 03_random_forest.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data_prep import load_data, report

train, test, FEATURES, TARGET = load_data()

model = RandomForestRegressor(
    n_estimators=300,      # number of trees
    min_samples_leaf=3,    # a leaf must cover at least 3 hours, which limits overfitting
    random_state=42,
    n_jobs=-1,
)
model.fit(train[FEATURES], train[TARGET])
predictions = model.predict(test[FEATURES])

importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Which features the forest relies on:")
for name, value in importance.items():
    print(f"  {name:<16} {value:.1%}")
print()

report("Random Forest", test[TARGET], predictions)
