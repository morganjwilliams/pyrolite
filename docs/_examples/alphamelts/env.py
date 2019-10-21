from pyrolite.ext.alphamelts.env import MELTS_Env
# %% Env
# set up an environment for an experiment stepping down temperature from 1350 to 1000
# in 3 degree steps
env = MELTS_Env()
env.MINP = 7000
env.MAXP = 7000
env.MINT = 1000
env.MAXT = 1350
env.DELTAT = -3
# %% save file
# you can directly export these parameters to an envrionment file here:
with open("pyrolite_envfile.txt", "w") as f:
    f.write(env.to_envfile(unset_variables=False))
# then pass this to alphamelts using run_alphamelts.command -f pyrolite_envfile.txt
