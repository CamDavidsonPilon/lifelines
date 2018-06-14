import numpy as np
import seaborn as sns
sns.set()

### avoid coding problems ####
import sys 
reload(sys)
sys.setdefaultencoding('gbk')
##############################

# load data
from lifelines.datasets import load_regression_dataset
regression_dataset = load_regression_dataset()
#print regression_dataset

# fit
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(regression_dataset, 'T', event_col='E')
X = regression_dataset.drop(['E', 'T'], axis=1)

# draw
cph.predict_survival_function(X.iloc[1:5]).plot()
import matplotlib.pyplot as plt
plt.xlim(0, 10)
plt.ylim(0.2, 1)
plt.title('survival curves for events')
plt.show()
