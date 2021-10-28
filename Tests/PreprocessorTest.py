from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp

vk_data = pd.read_csv("../Data/vk_data.csv")
print(vk_data.mean_tr)

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

data, en = p.apply(vk_data)
# for i, j in en.items():
#     print(f"{i}:{j}")

print(p.get_info())

