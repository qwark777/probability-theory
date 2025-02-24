from ucimlrepo import fetch_ucirepo
import pandas as pd
automobile = fetch_ucirepo(id=10)
# price continuous написано на сайте https://archive.ics.uci.edu/dataset/10/automobile
data = pd.DataFrame(automobile.data)
data.to_csv('automobile_data.csv', index=False)
