import matplotlib.pyplot as plt
from pyrolite.util.classification import Geochemistry
from pyrolite.util.synthetic import test_df
import numpy as np

np.random.seed(41)
# create the classifier
cm = Geochemistry.TAS()

# create some random data
mean = np.array([49, 11, 15, 4, 0.5, 4, 1.5])
df = (
    test_df(
        cols=["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O"],
        mean=mean,
        index_length=100
    )
    * mean.sum()
)

# create a TotalAlkali column
df["TotalAlkali"] = df.Na2O + df.K2O
# classify
df["TAS"] = cm.classify(df)

# add the potential rock names
df["Rocknames"] = df.TAS.apply(
    lambda x: cm.clsf.fields.get(x, {"names": None})["names"]
)

# visualise the classification diagram
fig, ax = plt.subplots(1)

ax.scatter(df.SiO2, df.TotalAlkali, c="k", alpha=0.2)
cm.add_to_axes(ax, color="0.5", alpha=0.5, zorder=-1)
