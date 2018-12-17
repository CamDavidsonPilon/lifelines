from .coxph_fitter import CoxPHFitter
from ..utils import _get_index

__author__ = "KOLANICH"

from Chassis import Chassis
from AutoXGBoost import AutoXGBoost
import pandas
import typing

class XGBoostCoxPHFitter(CoxPHFitter):

	"""
	This class implements fitting Cox's proportional hazard model using XGBoost `cox:survival` objective contributed by @slundberg.
	This module uses some libraries like 
	"""

	def __init__(self, spec, hyperparams=None, alpha=0.95, durationColPostfix="_prep", prefix=None):
		if not (0 < alpha <= 1.):
			raise ValueError('alpha parameter must be between 0 and 1.')
		
		self.alpha = alpha
		self.initialSpec=spec
		self.hyperparams=hyperparams
		self._defaultDurationCol=None
		self.strata=None # to make CoxPHFitter happy
		self.prefix=prefix
		for k,v in spec.items():
			if v=="survival":
				self._defaultDurationCol=k
				break
	
	def prepareFitting(self, df, duration_col=None, event_col=None):
		self.spec=type(self.initialSpec)(self.initialSpec)
		
		df = df.copy()
		if duration_col:
			#df = df.sort_values(by=duration_col)
			pass
		
		duration_col_transformed = None
		if event_col is not None:
			if duration_col is None:
				if self._defaultDurationCol:
					duration_col_transformed=self._defaultDurationCol
			else:
				if self.spec[duration_col] == "survival":
					duration_col_transformed = duration_col
				elif self.spec[duration_col] in {"numerical", "stop"}:
					duration_col_transformed=duration_col+"_prep"
		
			if duration_col_transformed not in df.columns:
				df.loc[:, duration_col_transformed]=df.loc[:, duration_col]*(df.loc[:, event_col]*2-1)
			
			self.spec[duration_col] = "stop"
			self.spec[event_col] = "stop"
		else:
			assert duration_col is not None
			duration_col_transformed=duration_col
		
		self.duration_col_transformed=duration_col_transformed
		self.spec[duration_col_transformed] = "survival"
		
		#print(df)
		return AutoXGBoost(self.spec, df, prefix=self.prefix)
	
	def optimizeHyperparams(self, df, duration_col=None, event_col=None, show_progress=False, autoSave:bool=True, folds:int=10, iters:int=1000, jobs:int=None, optimizer:'UniOpt.core.Optimizer'=None, force:typing.Optional[bool]=None, *args, **kwargs):
		self.axg=self.prepareFitting(df, duration_col=duration_col, event_col=event_col)
		self.axg.optimizeHyperparams(columns={self.duration_col_transformed}, autoSave=autoSave, folds=folds, iters=iters, jobs=jobs, optimizer=optimizer, force=force, *args, **kwargs)

	def fit(self, df, duration_col=None, event_col=None, show_progress=True, initial_beta=None, saveLoadModel=None):
		"""
		Fit the XGBoost Cox Propertional Hazard model to a dataset.

		Parameters:
		  df: a Pandas dataframe with necessary columns `duration_col` and
			 `event_col`, plus other covariates. `duration_col` refers to
			 the lifetimes of the subjects. `event_col` refers to whether
			 the 'death' events was observed: 1 if observed, 0 else (censored).
		  duration_col: the column in dataframe that  contains the subjects'
			 lifetimes.
		  event_col: the column in dataframe that contains the subjects' death
			 observation. If left as None, assume all individuals are non-censored.
		  show_progress: since the fitter is iterative, show convergence
			 diagnostics.
		  initial_beta: initialize the starting point of the iterative
			 algorithm. Default is the zero vector.
		  step_size: set an initial step size for the fitting algorithm.

		Returns:
			self, with additional properties: hazards_

		"""
		
		self.axg=self.prepareFitting(df, duration_col=duration_col, event_col=event_col)
		assert self.duration_col_transformed
		
		if self.hyperparams is not None:
			self.axg.bestHyperparams=self.hyperparams
		else:
			self.axg.loadHyperparams()
		
		if saveLoadModel is True:
			self.axg.loadModel(cn=self.duration_col_transformed)
		
		#print(df[self.duration_col_transformed])
		self.axg.trainModels((self.duration_col_transformed,))
		
		if saveLoadModel is False:
			f.axg.models[self.duration_col_transformed].save()
		
		#self.hazards_ = 
		#self.confidence_intervals_ = self._compute_confidence_intervals()

		E=self.axg.select(columns={event_col})[event_col]
		T=self.axg.select(columns={duration_col})[duration_col]
		
		self.baseline_hazard_ = self._compute_baseline_hazards(self.axg.prepareCovariates("days_prep"), T, E)
		self.baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard()
		#self.baseline_survival_ = self._compute_baseline_survival()
		#self.score_ = concordance_index(self.durations, -self.baseline_survival_, self.event_observed)
		#self._train_log_partial_hazard = self.predict_log_partial_hazard(self._norm_mean.to_frame().T)
		return self
	
	def predict_log_partial_hazard(self, X):
		if not isinstance(X, Chassis):
			dmat=AutoXGBoost(self.spec, X)
		else:
			dmat=X
		res=self.axg.predict(self.duration_col_transformed, dmat, returnPandas=True)
		res.name=0
		return pandas.DataFrame(res)

