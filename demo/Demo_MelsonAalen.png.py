import seaborn as sns
sns.set()

### avoid coding problems ####
import sys 
reload(sys)
sys.setdefaultencoding('gbk')
##############################

from lifelines.datasets import load_waltons
df = load_waltons()
T = df['T']
E = df['E']

# fit
from lifelines import NelsonAalenFitter
naf = NelsonAalenFitter()

# draw
naf.fit(T, event_observed=E)
naf.plot()
import matplotlib.pyplot as plt
plt.legend(frameon=True)
plt.show()