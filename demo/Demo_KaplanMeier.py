import seaborn as sns
sns.set()

### avoid coding problems ####
import sys 
reload(sys)
sys.setdefaultencoding('gbk')
##############################

# load data
from lifelines.datasets import load_waltons
df = load_waltons()
T = df['T']
E = df['E']
#print df

# fit
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

# draw
groups = df['group']
ix = (groups == 'miR-137')
kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()
kmf.fit(T[ix], E[ix], label='miR-137')
kmf.plot(ax=ax,color='red')

import matplotlib.pyplot as plt
plt.legend(frameon=True)
plt.show()
