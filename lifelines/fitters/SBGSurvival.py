from __future__ import print_function
from lifelines.utils.DataHandler import DataHandler
from lifelines.utils.ShiftedBetaGeometric import ShiftedBetaGeometric
import numpy as np
import pandas as pd


class SBGSurvival(object):
    """
    This class implements an extended version of the Shifted-Beta-Geometric
    model by P. Fader and B. Hardie.

    The original model works by assuming a constant in time, beta distributed
    individual probability of churn. Due to the heterogeneity of a cohort's
    churn rates (since each individual will have a different probability of
    churning), expected behaviours such as the decrease of cohort churn rate
    over time arise naturally.

    The extension done here generalizes the coefficients alpha and beta of the
    original model to function of features on the individual level. A
    log-linear model is used to construct alpha(x) and beta(x) and the
    likelihood is then computed by combining the contributions of each and
    every sample in the training set.

    The model takes as inputs ...
    """

    def __init__(self,
                 age,
                 alive,
                 features=None,
                 gamma=1.0,
                 gamma_beta=None,
                 bias=True,
                 normalize=True,
                 verbose=False):
        """
        Initializes objects with parameters necessary to create the supporting
        objects: DataHandler and ShiftedBeta

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.
                --- See DataHandler.py

        :param alive: str
            The column name with the status of each individual. In the context
            of survival analysis, an individual may be dead or alive, and its
            contribution to the model will depend on it.
                --- See DataHandler.py

        :param features: str, list or None
            A string with the name of the column to be used as features, or a
            list of names of columns to be used as features or None, if no
            features are to be used.
                --- See DataHandler.py

        :param gamma: float
            A non-negative float specifying the strength of the regularization
            applied to w_alpha (alpha's weights) and, if gamma_beta is not
            given, it is also applied to beta.
                --- See ShiftedBeta.py

        :param gamma_beta: float
            A non-negative float specifying the strength of the regularization
            applied to w_beta (beta's weights). If specified, overwrites the
            value of gamma for beta.
                --- See ShiftedBeta.py

        :param bias: bool
            Whether or not a bias term should be added to the feature matrix.
                --- See DataHandler.py

        :param normalize: bool
            Whether or not numerical fields should be normalized (centered and
            scaled to have std=1)
                --- See DataHandler.py

        :param verbose: bool
            Whether of not status updates should be printed
                --- See ShiftedBeta.py
        """

        # Create objects!
        # DATA-HANDLER OBJECT
        # The DataHandler object may be created without the training data, so
        # we do it here.
        self.dh = DataHandler(age=age,
                              alive=alive,
                              features=features,
                              bias=bias,
                              normalize=normalize)

        # Shifted beta model object
        # Was a different gammab parameter passed? If not, we use the same
        # value passed to gamma.
        if gamma_beta is None:
            gamma_beta = 1.0 * gamma
        # create shifted beta object
        self.sb = ShiftedBetaGeometric(gamma_alpha=gamma,
                                       gamma_beta=gamma_beta,
                                       verbose=verbose)

    def fit(self, df, restarts=1):
        """
        A method responsible for learning both the transformation of the data,
        including addition of a bias parameters, centering and re-scaling of
        numerical features, and one-hot-encoding of categorical features. In
        addition to learning the parameters alpha and beta of the shifted-beta-
        geometric model.

        This is just a wrapper, the real heavy-lifting is done by the
        DataHandler and ShiftedBeta objects.

        :param df: pandas DataFrame
            A pandas DataFrame with similar schema as the one used to train
            the model. Similar in the sense that the columns used as cohort,
            age and categories must match. Extra columns with not affect
            anything.

        :param restarts: int
            Number of times to restart the optimization procedure with a
            different seed, to avoid getting stuck on local maxima.
        """
        # Transform dataframe extracting feature matrix, ages and alive status.
        x, y, z = self.dh.fit_transform(df)

        # fit to data using the ShiftedBeta object.
        self.sb.fit(X=x,
                    age=y,
                    alive=z,
                    restarts=restarts)

    def summary(self):
        """
        Simple method to get the learned weights and their corresponding
        categories

        :return: pandas DataFrame
            A DataFrame object with alpha and beta weights for each category
        """
        # Construct a DataFrame consisting of feature name and corresponding
        # alpha and beta parameters. Names are obtained by invoking the
        # get_names() method, and the parameter displayed are the weights,
        # not the final values (since that cannot be made sense in separate).
        suma = pd.DataFrame(data={name: (a, b) for name, a, b in
                                  zip(self.dh.get_names(),
                                      self.sb.alpha,
                                      self.sb.beta)},
                            index=['w_alpha', 'w_beta']
                            ).T
        return suma

    def predict_params(self, df):
        """
        predict_params is a method capable of predicting the values of alpha
        and beta for given combination of features. It invokes the
        compute_alpha_beta method from the ShiftedBeta object to compute the
        arrays of alpha and beta for every sample in df given the available
        features.

        Notice that it must first transform the dataframe df using
        DataHandler's transform method, so that it can than work with the lower
        level feature matrix, x.

        :param df: pandas DataFrame
            A pandas dataframe with at least the same feature columns as the
            one used to train the model.

        :return: pandas DataFrame
            A DataFrame with the predicted alpha and beta for each sample in df
        """
        # Start by transforming df to its lower level np.array representation
        x, y, z = self.dh.transform(df=df)

        # Use compute_alpha_beta to compute alpha and beta for every sample in
        # df based on the feature matrix extracted from df, x.
        alpha, beta = self.sb._compute_alpha_beta(x, self.sb.alpha, self.sb.beta)

        # Return a dataframe with predictions.
        return pd.DataFrame(data=np.vstack([alpha, beta]),
                            index=['alpha', 'beta']).T

    def predict_churn(self, df, age=None, **kwargs):
        """
        predict_churn is a method to compute churn rate for a number of periods
        conditioned on the age of the sample.

        This method invokes the churn_p_of_t method from ShiftedBeta to compute
        the churn rate for a given number of periods conditional on age. See
        the description of churn_p_of_t in ShiftedBeta.py for more details.

        This method is a wrapper, it transforms the dataframe df to the
        appropriate representation and feed it to the lower level method from
        ShiftedBeta.

        It is worth noticing that the user has the option to pass the value for
        age, which can wither be a single number of an array with the same
        length as df, and this will overwrite whatever other value for age
        might come out when transforming df.

        :param df: pandas DataFrame
            A pandas dataframe with at least the same feature columns as the
            one used to train the model.

        :param age: None or float or ndarray of shape(df.shape[0], )
            If age is None, the method will use the age parameter extracted
            from df.
            ** Notice that if age=None and df does not contain an age field,
            a RuntimeError will be raised! **
            If age != None, pass this value along to churn_p_of_t.

        :param kwargs:
            Any other arguments that should be redirected to churn_p_of_t.

        :return: pandas DataFrame
            A DataFrame with the churn_p_of_t matrix.
        """
        x, y, z = self.dh.transform(df=df)

        # If age field is present in prediction dataframe, we may choose to
        # use it to calculate future churn. To do so, we first check if the
        # user passed a new age parameter, if answer is yes, use the new age.
        # If, however, the user did not pass age, use the value extracted from
        # the dataframe, df.
        # ** If no value for age is passed and the dataframe does not contain
        # age, a RuntimeError is raised.
        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        # Create a dataframe with the churn_p_of_t matrix with all relevant
        # parameters.
        out = pd.DataFrame(data=self.sb.churn_p_of_t(x, age=age, **kwargs))

        # Give columns a decent, generic name.
        out.columns = ['period_{}'.format(col)
                       for col in range(1, out.shape[1] + 1)]

        return out

    def predict_survival(self, df, age=None, **kwargs):
        """
        predict_survival is a method to compute the survival curve for a number
        of periods conditioned on the age of the sample.

        This method invokes the survival_function method from ShiftedBeta to
        compute the retention rate for a given number of periods conditional
        on age. See the description of survival_function in ShiftedBeta.py for
        more details.

        This method is a wrapper, it transforms the dataframe df to the
        appropriate representation and feed it to the lower level method from
        ShiftedBeta.

        It is worth noticing that the user has the option to pass the value for
        age, which can wither be a single number of an array with the same
        length as df, and this will overwrite whatever other value for age
        might come out when transforming df.

        :param df: pandas DataFrame
            A pandas dataframe with at least the same feature columns as the
            one used to train the model.

        :param age: None or float or ndarray of shape(df.shape[0], )
            If age is None, the method will use the age parameter extracted
            from df.
            ** Notice that if age=None and df does not contain an age field,
            a RuntimeError will be raised! **
            If age != None, pass this value along to survival_function.

        :param kwargs:
            Any other arguments that should be redirected to survival_function.

        :return: pandas DataFrame
            A DataFrame with the survival_function matrix.
        """
        x, y, z = self.dh.transform(df=df)

        # If age field is present in prediction dataframe, we may choose to
        # use it to calculate future churn. To do so, we first check if the
        # user passed a new age parameter, if answer is yes, use the new age.
        # If, however, the user did not pass age, use the value extracted from
        # the dataframe, df.
        # ** If no value for age is passed and the dataframe does not contain
        # age, a RuntimeError is raised.
        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        # Create a dataframe with the churn_p_of_t matrix with all relevant
        # parameters.
        out = pd.DataFrame(data=self.sb.survival_function(x,
                                                          age=age,
                                                          **kwargs))

        # Give columns a decent, generic name.
        out.columns = ['period_{}'.format(col)
                       for col in range(1, out.shape[1] + 1)]

        return out

    def predict_ltv(self, df, age=None, alive=None, **kwargs):
        """
        predict_ltv is a method to compute the ltv for each sample conditioned
        on age.

        This method invokes the derl method from ShiftedBeta to compute
        the residual ltv of each sample given its given age. See the
        description of derl in ShiftedBeta.py for more details.

        This method is a wrapper, it transforms the dataframe df to the
        appropriate representation and feed it to the lower level method from
        ShiftedBeta.

        It is worth noticing that the user has the option to pass the value for
        both age and alive fields, which can wither be a single number of an
        array with the same length as df, and this will overwrite whatever
        other value for age and/or alive might come out when transforming df.

        :param df: pandas DataFrame
            A pandas dataframe with at least the same feature columns as the
            one used to train the model.

        :param age: None or float or ndarray of shape(df.shape[0], )
            If age is None, the method will use the age parameter extracted
            from df.
            ** Notice that if age=None and df does not contain an age field,
            a RuntimeError will be raised! **
            If age != None, pass this value along to derl.

        :param alive: None or float or ndarray of shape(df.shape[0], )
            If age is None, the method will use the alive parameter extracted
            from df.
            ** Notice that if alive=None and df does not contain an alive
            field, a RuntimeError will be raised! **
            If alive != None, pass this value along to derl.

        :param kwargs:
            Any other arguments that should be redirected to derl.

        :return: pandas DataFrame
            A DataFrame with the ltv predictions.
        """
        x, y, z = self.dh.transform(df=df)

        # If age field is present in prediction dataframe, we may choose to
        # use it to calculate future churn. To do so, we first check if the
        # user passed a new age parameter, if answer is yes, use the new age.
        # If, however, the user did not pass age, use the value extracted from
        # the dataframe, df.
        # ** If no value for age is passed and the dataframe does not contain
        # age, a RuntimeError is raised.
        if age is None:
            age = y
        if age is None:
            raise RuntimeError('The "age" field must either be present in '
                               'the dataframe or passed separately as an '
                               'argument.')

        # See the discussion above for age, exact same logic applies.
        if alive is None:
            alive = z
        if alive is None:
            raise RuntimeError('The "alive" must either be present in the '
                               'dataframe or passed separately as an '
                               'argument.')

        # Get LTVs and return a dataframe!
        ltvs = self.sb.derl(x, age=age, alive=alive, **kwargs)

        return pd.DataFrame(data=ltvs, columns=['ltv'])
