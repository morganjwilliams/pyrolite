import matplotlib.pyplot as plt
from pyrolite.classification import Geochemistry
from pyrolite.util.pd import test_df
import numpy as np

np.random.seed(110)
# create the classifier
cm = Geochemistry.TAS()

# create some random data
df = test_df(cols=['SiO2', 'Na2O', 'K2O'], index_length=50)
df.SiO2 *= 70
df.K2O *= 10
# create a TotalAlkali column
df['TotalAlkali'] = df.Na2O + df.K2O
# classify
df['TAS'] = cm.classify(df)

# add the potential rock names
df['Rocknames'] = df.TAS.apply(lambda x:
                              cm.clsf.fields.get(x, {'names': None})['names'])

# visualise the classification diagram
fig, ax = plt.subplots(1)
ax.scatter(df.SiO2, df.TotalAlkali, c='k')
cm.add_to_axes(ax, color='0.5', alpha=0.5, zorder=-1)
