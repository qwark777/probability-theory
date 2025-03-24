import pandas as pd
from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets
data = pd.concat([X, y], axis=1)
data.to_csv('vine.csv', index=False)

