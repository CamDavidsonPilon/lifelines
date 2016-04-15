from __future__ import print_function
from scipy.optimize import minimize
from math import log10
from datetime import datetime
from scipy.special import hyp2f1, betaln
import numpy as np


class ShiftedBetaGeometric(object):
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

    The model takes as inputs a feature matrix X, an (int) array of ages and
    a boolean array indicating whether or not each individual is still active.
    Therefore this object works similarly to any other survival analysis tool,
    with the SBS model as the underlying model.
    """

    def __init__(self,
                 gamma_alpha=1.0,
                 gamma_beta=1.0,
                 verbose=False):
        """
        This object is initialized with training time hyper-parameters and a
        verbose option.

        :param gamma_alpha: float
            A non-negative float specifying the strength of the regularization
            applied to w_alpha (alpha's weights).

        :param gamma_beta: float
            A non-negative float specifying the strength of the regularization
            applied to w_beta (beta's weights).

        :param verbose: bool
            Whether of not status updates should be printed
        """

        # --- Parameters ---
        # alpha and beta are the parameters learned by this model. When the
        # time is right they will be arrays with length a function of number of
        # predictors and whether or not a bias is being used. For now we
        # create a place holder array of a single zero.#
        self.alpha = np.zeros(1)
        self.beta = np.zeros(1)

        # --- Regularization ---
        # In this model regularization helps by both limiting the model's
        # complexity as well as greatly improving numerical stability during
        # the optimization process.
        # Moreover different regularization parameters for alpha and beta
        # can be helpful, specially in extreme cases when the distribution
        # is near the extremes (0 or 1).

        # Clearly both gammas must be non-negative, so we make sure to check
        # for it here.
        if gamma_alpha < 0:
            raise ValueError("The regularization constant gamma_alpha must "
                             "be a non-negative real number. A negative "
                             "value of {} was passed.".format(gamma_alpha))

        if gamma_beta < 0:
            raise ValueError("The regularization constant gamma_beta must "
                             "be a non-negative real number. A negative "
                             "value of {} was passed.".format(gamma_beta))

        self.gammaa = gamma_alpha
        self.gammab = gamma_beta

        # Boolean variable controlling whether or not status updates should be
        # printed during training and other stages.
        self.verbose = verbose

        # A variable to store the size of the dataset, useful in certain
        # situations where no predictors are being used.
        self.n_samples = 0

    @staticmethod
    def _log_retention_stats(alpha, beta, num_periods):
        """
        A function to calculate the expected probabilities recursively.
        Using equation 7 from [1] and the alpha and beta coefficients
        obtained by training this model, it computes P(T = t)  as well
        as S(T = t) recursively, returning only the relevant values
        for computing the individual contribution to the likelihood.

        :param alpha: np.array
            A value for the alpha parameter.

        :param beta: np.array
            A value for the beta parameter.

        :param num_periods: Int
            The number of periods for which the probability of churning
            should be computed.

        :return: (float, float)
            A tuple with both the probability of dieing as well as
            surviving the current period.
        """
        # Extreme values of alpha and beta can cause severe numerical stability
        # problems! We avoid some of if by clipping the values of both alpha
        # and beta parameters such that they lie between 1e-5 and 1e5.
        alpha = np.clip(alpha, 1e-5, 1e5)
        beta = np.clip(beta, 1e-5, 1e5)

        # Use death probability of the most recent period
        p = betaln(alpha + 1, beta + num_periods - 1)
        p -= betaln(alpha, beta)

        # Use survival value from previous period, hence the minus one
        s = betaln(alpha, beta + num_periods - 1)
        s -= betaln(alpha, beta)

        # Note that p_new is the likelihood of not making to the next period
        # while s_old is the likelihood of surviving the current period. Which
        # is used in the likelihood depends on whether or not the subject is
        # alive or dead.
        return p, s

    @staticmethod
    def _compute_alpha_beta(X, w_a, w_b):
        """
        This method computes the float values of alpha and beta given a matrix
        of predictors X and an array of weighs wa and wb. It does so by taking
        the dot product of w_a (w_b) with the matrix X and exponentiating it
        the resulting array.

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix.

        :param w_a: ndarray of shape (n_features, )
            Array of weights for the alpha parameter

        :param w_b: ndarray of shape (n_features, )
            Array of weights for the alpha parameter

        :return: (ndarray, ndarray) both with shapes (n_samples, )
            The alpha and beta values calculated for each row of the feature
            matrix X.
        """

        # Take that dot product!
        waT_dot_X = (w_a * X).sum(axis=1)
        wbT_dot_X = (w_b * X).sum(axis=1)

        # Return the element-wise exponential
        return np.exp(waT_dot_X), np.exp(wbT_dot_X)

    def _logp(self, X, age, alive, wa, wb):
        """
        The LogLikelihood function. Given the data and relevant
        variables this function computed the loglikelihood.

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix.

        :param age: ndarray of shape (n_samples, )
            Integer array of ages for each samples in the dataset.

        :param alive: ndarray of shape (n_samples, )
            Binary array indicating whether or not a particular sample is
            active or not.

        :param wa: ndrray of shape (n_features, )
            The weights used to construct the alpha parameter.

        :param wb: ndrray of shape (n_features, )
            The weights used to construct the beta parameter.

        :return: float
            Negative value of the loglikelihood
        """
        # --- LogLikelihood (One Cohort at a Time) --- #
        # We calculate the LogLikelihood for each cohort separately and
        # combining them. From appendix B in [1] it is easy to see that
        # the extension of the model to multiple cohorts of different
        # sizes is simply given a similar product as in B1, except that
        # each month of each cohort will contribute with a term like:
        #       P(T = t | alpha, beta) ** n_t
        # Which, when taking the log, translates to a sum similar to B3,
        # but extended to include all cohorts.
        log_like = 0.0

        # L2 regularizer
        # In this model regularization helps by both limiting the model's
        # complexity as well as greatly improving numerical stability during
        # the optimization process.
        # However, it is undesirable to regularize the bias weights, since
        # this can stop the model from learning anything. *** Note that
        # this is different than the case of, say, a linear model, where the
        # trivial model (with zero weights) approximates the mean of the
        # target values. Here, the absence of weights (including bias) does
        # NOT lead to a trivial model, but one with a unreasonable
        # preference for alpha = exp(0) and beta = exp(0). ***
        # Moreover different regularization parameters for alpha and beta
        # can be helpful, specially in extreme cases when the distribution
        # is near the extremes (0 or 1)
        l2_reg = self.gammaa * sum(wa[1:]**2) + self.gammab * sum(wb[1:]**2)

        # update log-likelihood with regularization val.
        log_like -= l2_reg

        # Get the full alpha and beta for each sample and store the values in
        # ndarrays of shape (n_samples, )
        alpha, beta = self._compute_alpha_beta(X, wa, wb)

        # loop over data and add the contribution to the log-likelihood from
        # each sample in the dataset.
        # Notice that once we have alpha and beta for each sample we do not
        # need the feature matrix to compute the contribution to the
        # log-likelihood.
        retention_probs = self._log_retention_stats(alpha, beta, age)

        # Contribution to log likelihood based on status (alive vs. dead)
        log_like += np.sum(retention_probs[0][np.where(alive == 0)[0]])
        log_like += np.sum(retention_probs[1][np.where(alive == 1)[0]])

        # Negative log_like since we will use scipy's minimize object.
        return -log_like

    def fit(self, X, age, alive, restarts=1):
        """
        Method responsible for the learning step it takes all the relevant data
        as argument as well as the number of restarts with random seeds to
        perform. While restarting with other seeds sounds like a good idea the
        model has proven to be fairly stable and this may be removed in the
        future.

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix

        :param age: ndarray of shape (n_samples, )
            An array with the age of each individual.

        :param alive: ndarray of shape (n_samples, )
            An array with

        :param restarts: int
            Number of times to run the optimization procedure with different seeds
        """

        # Make sure ages are all non negative...
        min_age = min(age)
        if min(age) < 1:
            raise ValueError("All values of age must be equal or greater to "
                             "one. The minimum value of "
                             "{} was found.".format(min_age))

        # Make sure alive is either zero or one!
        alive_vals = set(alive)
        if alive_vals != {0, 1}:
            raise ValueError('Values for alive must be either zero or one. A '
                             'value of '
                             '{} was found.'.format(list(alive_vals - {0, 1})))

        # The amount of free space to create when printing based on the total
        # amount of restarts.
        print_space = int(max(int(log10(restarts)) + 1, 5))

        # store the number of samples we are dealing with
        self.n_samples = age.shape[0]

        # Now that we have X we can calculate the number of parameters we need
        n_params = X.shape[1]

        # --- Optimization Starting Points
        # Generate random starting points for optimization step.
        initial_guesses = 0.1 * np.random.randn(restarts, 2 * n_params) - 0.01

        # Initialize optimal value to None
        # I choose not to set it to, say, zero, or any other number, since I am
        # not sure that the log-likelihood is bounded in anyway. So is better
        # to initialize with None and use the first optimal value to get the
        # ball rolling.
        optimal = None

        # To comply with scipy's optimization object, alpha and beta are
        # concatenated and passed as one single array of shape 2 * n_params.
        opt_params = np.zeros((2 * n_params))

        # clock
        start = datetime.now()

        # Print a nice looking  header for the optimization process
        if self.verbose:
            print('Starting Optimization with parameters:')
            print('{:>15}: {}'.format('Samples', self.n_samples))
            print('{:>15}: {}'.format('gamma (alpha)', self.gammaa))
            print('{:>15}: {}'.format('gamma (beta)', self.gammaa))
            print('{:>15}: {}'.format('Seeds', restarts))

            print()
            print("{0:^{3}} | {1:^10} | {2:13} |".format('Step',
                                                         'Time',
                                                         'LogLikelihood',
                                                         print_space))
            print("-"*35)

        # Run likelihood optimization "restarts" times.
        # noinspection PyTypeChecker
        for step, guess in enumerate(initial_guesses):
            # --- Variables
            #   step: Integer - current step number
            #  guess: Array - array with starting points for minimization

            # --- Optimization
            # Unbounded optimization (minimization) of negative log-likelihood
            # Use a lambda function to make explicit the choice of variables.
            # Notice that the variable p is an array of length 2 * n_params,
            # which is split in half when passed to the _logp method. First
            # half is alpha, second is beta.
            new_opt = minimize(lambda p: self._logp(X=X,
                                                    age=age,
                                                    alive=alive,
                                                    wa=p[:n_params],
                                                    wb=p[n_params:]),
                               guess,
                               bounds=[(None, None)] * n_params * 2
                               )

            # For the first run only optimal is None, that being the case, we
            # set the current values to both optimal (function value) as well
            # as opt_params - parameters that minimize the function.
            if optimal is None:
                optimal = new_opt.fun
                opt_params = new_opt.x

            # Have we found a better value yet? If we have, update optimal
            # with new function minimum and opt_params with corresponding
            # minimizing parameters.
            if new_opt.fun < optimal:
                optimal = new_opt.fun
                opt_params = new_opt.x

            # Print current status if verbose is True.
            if self.verbose:
                print_string = "{step: {space}} | {elap_time:^10} | {func:13.5f} |"
                print(print_string.format(step=step + 1,
                                          elap_time=str(datetime.now() - start)[:-7],
                                          func=float(optimal),
                                          space=print_space))

        # --- Parameter Values
        # Optimization is complete, time to save best parameters.
        # Note that we breakdown the parameters passed to and returned from
        # the scipy.optimize.minimize object in two. The first half correspond
        # to the alpha parameter, while the second half is beta.
        self.alpha = opt_params[:n_params]
        self.beta = opt_params[n_params:]

        # --- Regularization Penalty
        # Compute the regularization penalty applied to the parameter vectors.
        # Remember that there is no penalty for bias!
        reg_penalty = self.gammaa * sum(self.alpha[1:]**2) + \
            self.gammab * sum(self.beta[1:]**2)

        # Print some final remarks before we say goodbye.
        if self.verbose:
            print()
            print('Optimization completed:')
            print('{:>15}: {}'.format('wa', self.alpha))
            print('{:>15}: {}'.format('wb', self.beta))
            print('{:>15}: {}'.format('LogLikelihood', optimal))
            print('{:>15}: {}'.format('Reg. Penalty', reg_penalty))
            print()

    def derl(self,
             X,
             age=1,
             alive=1,
             arpu=1.0,
             discount_rate=0.005):
        """
        Discounted Expected Residual Lifetime, as derived in [2].
        See equation (6).

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix

        :param age: ndarray of shape (n_samples, )
            An array with the age of each individual.

        :param alive: ndarray of shape (n_samples, )
            An array with

        :param arpu: float or ndarray of shape(n_samples, )
            The average revenue per user. It can either be a float, in which
            case the same value is used for all entries, or an array matching
            the shape of age, in which case a different value is applied to
            each data sample.

        :param discount_rate: float
            The discount rate to be used. Must be a positive real number.

        :return: ndarray of shape (n_samples, )
            Returns the DERL - Discounted Expected Residual Lifetime for each
            sample in X. Additionally it sets to zero the DERL for any sample
            that is not active (alive = 0).
        """
        # only positive ages, please
        try:
            min_age = min(age)
        except TypeError:
            min_age = 1.0 * age
        finally:
            # age cannot be negative!
            if min_age < 1:
                raise ValueError("All ages must be positive. A value of "
                                 "{} was passed.".format(min_age))
            del min_age

        # Ensure the discount rate makes sense. It must be a positive real
        # number.
        if discount_rate <= 0:
            raise ValueError("The discount rate must be a positive real "
                             "number. A value of "
                             "{} was passed.".format(discount_rate))

        # Compute the full alpha and beta parameters for all samples present in
        # the dataset X. Uses the instance variables self.alpha and self.beta
        # that store the fitted values for weights wa and wb used in the linear
        # model that sits on top of the SBS model.
        alpha, beta = self._compute_alpha_beta(X, self.alpha, self.beta)

        # To make it so that the formula resembles that of the paper we define
        # the parameter n as below.
        # In the paper n is related to the n-th contract period, starting at
        # one. And all calculations are done with respect to the moment right
        # before the end of such period, the moment prior to the customer
        # making the decision to renew his membership. Which matches precisely
        # the notion of age used throughout.
        n = age

        # The equation is two long, so we break in two parts.
        f1 = (beta + n - 1) / (alpha + beta + n - 1)
        f2 = hyp2f1(1., beta + n, alpha + beta + n, 1. / (1. + discount_rate))

        # DERL is given by f1 * f1. In addition to it we use the arpu and the
        # alive field to create the residual LTV, which is the final output of
        # this method.
        return arpu * f1 * f2 * alive

    def churn_p_of_t(self,
                     X,
                     age=1,
                     n_periods=12):
        """
        churn_p_of_t computes the churn as a function of time curve. Using
        equation 7 from [1] and the alpha and beta coefficients obtained by
        training this model, it computes P(T = t) recursively, returning either
        the expected value or an array of values.

        ** Notice that the churn(t) curve is calculated starting at the age
        value passed. There is, if age=10 and n_periods = 5 the churn(t)
        values will be those of months 11 to 15. **

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix

        :param age: ndarray of shape (n_samples, )
            An array with the age of each individual.

        :param n_periods: int
            The number of periods for which churn is to be calculated.

        :return: ndarray of shape (n_samples, n_periods)
            Matrix with expected churn for each period following the age
            value per sample.

            There is, if n_periods = 5 and age = [1, 2, 3], the churn
            periods returned will be:

            output = [[c2, c3, c4, c5, c6],
                      [c3, c4, c5, c6, c7],
                      [c4, c5, c6, c7, c8]]
        """

        # Spot check making sure the values passed for n_periods make sense!
        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

        # Load alpha and beta sampled from the posterior. These fully
        # determine the beta distribution governing the customer level
        # churn rates
        alpha, beta = self._compute_alpha_beta(X, self.alpha, self.beta)

        # set the number of samples
        n_samples = X.shape[0]

        # Make sure age is an array of length X.shape[0]
        try:
            len(age) == n_samples
            # If no error, make sure age is ndarray
            age = np.array(age)
        except TypeError:
            # If age was passed as a number (float or int) we must turn it into
            # an array of the correct shape and repeated values.
            age = age * np.ones(n_samples, dtype=int)
        finally:
            # If we got here it means age is already a np.array, so we make
            # sure it has the correct number of entries.
            if age.shape[0] != n_samples:
                raise ValueError("Age must either be a number of an array with"
                                 " the same length as X.")

        # Now that we have age as a nice np.array we make sure all values are
        # non-negative!
        if min(age) < 0:
            raise ValueError("All ages must be non-negative.")

        # --- Initialize Output ---
        # Initialize the output as a matrix of zeros. The number of rows is
        # given by the total number of samples, while the number of columns
        # is given by the maximum value of age + n_periods.
        # The reason we start with this shape is that it is simpler to compute
        # churn for all possible months up to max(age) + n_periods and the
        # choose only the relevant ones to output.
        p_churn_matrix = np.zeros((n_samples, max(age) + n_periods))

        # Given the discussion above we also initialize a matrix to keep track
        # of which samples from the p_churn_matrix matrix should be returned to
        # the user. This is a boolean matrix of same shape initialized to false
        # that is updated as we go.
        # The idea is, for a given row in output we have False in all columns
        # representing periods prior to the age of that row and post the number
        # of periods + age for that row.
        # As we loop through and construct the churn(t) matrix we also make
        # sure to indicate which row should be turned on or off.
        outputs = np.zeros((n_samples, max(age) + n_periods), dtype=bool)

        # We start by setting to true all entries of column zero for which age
        # is zero. That means the output for entries for which age=0 will start
        # at the very first column, which is the very first value of churn
        # (zero).
        outputs[:, 0][np.where(age < 1)[0]] = True

        # Second step is to turn on the second column fo the output matrix for
        # entries with age < 2.
        # The reasoning is similar as above but not only we start the output
        # for entries with age = 1, but we also continue the output for entries
        # with age = 0.
        outputs[:, 1][np.where(age < 2)[0]] = True

        # --- Fill output recursively (see eq.7 in [1]) ---

        # Start with month one (churn rate of month zero was set to 0 by
        # definition).
        p_churn_matrix[:, 1] = alpha / (alpha + beta)

        # Calculate remaining months recursively using the formulas
        # provided in the original paper.
        for period in range(2, max(age) + n_periods):

            month = period
            update = (beta + month - 2) / (alpha + beta + month - 1)

            # None that i + 1 is simply the previous value, since val
            # starts with the third entry in the array, but I starts
            # counting form zero!
            p_churn_matrix[:, period] += update * p_churn_matrix[:, period - 1]

            # Turn on the relevant rows using the logic explained above.
            rows = np.where((period >= age) & (period < (age + n_periods)))[0]
            outputs[:, period][rows] = True

        # Return only the appropriate values and reshape the matrix.
        return p_churn_matrix[outputs].reshape((n_samples, n_periods))

    def survival_function(self, X, age=1,  n_periods=12):
        """
        survival_function computes the survival curve obtained from the model's
        parameters and assumptions. Using equation 7 from [1] and the alpha and
        beta coefficients obtained by training this model, it computes S(T = t)
        recursively, returning either the expected value or an array of values.
        To do so it must first invoke the self.churn_p_of_t method to calculate
        the monthly churn rates for the given time window, and then use it to
        compute the survival curve recursively.

         ** Notice that the churn(t) curve is calculated starting at the age
        value passed. There is, if age=10 and n_periods = 5 the churn(t)
        values will be those of months 11 to 15. **

        :param X: ndarray of shape (n_samples, n_features)
            The feature matrix

        :param age: ndarray of shape (n_samples, )
            An array with the age of each individual.

        :param n_periods: int
            The number of periods for which churn is to be calculated.

        :return: ndarray of shape (n_samples, n_periods)
            Matrix with expected retention rates for each period following the
            age value per sample.

            There is, if n_periods = 5 and age = [1, 2, 3], the churn
            periods returned will be:

            output = [[c2, c3, c4, c5, c6],
                      [c3, c4, c5, c6, c7],
                      [c4, c5, c6, c7, c8]]
        """

        # Spot check making sure the values passed make sense!
        if n_periods < 0 or not isinstance(n_periods, int):
            raise ValueError("The number of periods must be a non-zero "
                             "integer")

        # --- Churn Rates ---
        # Start by calling the method churn_p_of_t to calculate the monthly
        # churn rates gives the model's fitted parameters and the parameters
        # passed to the function.
        # Since this method is able to return the normalized tail of the
        # survival function (the retention rates using other than the
        # starting point as normalization), we must make sure we compute the
        # churn rates for a long enough period. Therefore we pass the value
        # n_months + renewals to the n_month parameter of the churn_p_of_t
        # which guarantees the churn rate curve extends far enough into the
        # future.#

        # set the number of samples
        n_samples = X.shape[0]

        # get number of periods
        try:
            # is age a list like object?
            len(age) == n_samples

            # If no error, make sure age is ndarray
            age = np.array(age)
        except TypeError:
            # If age was passed as a number (float or int) we must turn it into
            # an array of the correct shape and repeated values.
            age = age * np.ones(n_samples, dtype=int)
        finally:
            # Set num_periods the the sum of max(age) and the parameter
            # n_periods. Notice that, while not ideal, age may not be an
            # integer! While the model does not account for that
            # explicitly, and does not make proper use of it, it will work! For
            # this reason it is important to make sure num_periods is an
            # integer.
            num_periods = int(max(age) + n_periods)

            # If we got here it means age is already a np.array, so we make
            # sure it has the correct number of entries.
            if age.shape[0] != n_samples:
                raise ValueError("Age must either be a number of an array with"
                                 " the same length as X.")

        # Now that we have age as a nice np.array we make sure all values are
        # non-negative!
        if min(age) < 0:
            raise ValueError("All ages must be non-negative.")

        # Age of zero means we want survival func of current month as the
        # starting point! We do so to make sure we have all the necessary
        # information to construct the retention curve recursively.
        p_of_t = self.churn_p_of_t(X=X,
                                   age=0,
                                   n_periods=num_periods)

        # In a similar fashion as was done above in the churn_p_of_t method, we
        # will create two matrices.
        # One, s, will hold all values of the retention curve, starting from
        # period zero.
        # The other, outputs, is a boolean matrix indicating with entries of
        # the matrix s will be used as output.
        s = np.zeros(p_of_t.shape)

        # By definition, the very first column of s is one for all entries.
        s[:, 0] = 1.

        # Given the discussion above we also initialize a matrix to keep track
        # of which samples from the p_churn_matrix matrix should be returned to
        # the user. This is a boolean matrix of same shape initialized to false
        # that is updated as we go.
        # The idea is, for a given row in output we have False in all columns
        # representing periods prior to the age of that row and post the number
        # of periods + age for that row.
        # As we loop through and construct the churn(t) matrix we also make
        # sure to indicate which row should be turned on or off.
        outputs = np.zeros(p_of_t.shape, dtype=bool)

        # We start by setting to true all entries of column zero for which age
        # is zero. That means the output for entries for which age=0 will start
        # at the very first column, which is the very first value of churn
        # (zero).
        outputs[:, 0][np.where(age < 1)[0]] = True

        # Second step is to turn on the second column fo the output matrix for
        # entries with age < 2.
        # The reasoning is similar as above but not only we start the output
        # for entries with age = 1, but we also continue the output for entries
        # with age = 0.
        outputs[:, 1][np.where(age < 2)[0]] = True

        # --- Fill output recursively (see eq.7 in [1]) ---
        # Calculate remaining months recursively using the formulas
        # provided in the original paper.
        for col in range(1, s.shape[1]):
            s[:, col] = s[:, col - 1] - p_of_t[:, col]

            # Turn on the relevant rows using the logic explained above.
            rows = np.where((col >= age) & (col < (age + n_periods)))[0]
            outputs[:, col][rows] = True

        # pick correct entries
        s = s[outputs].reshape((n_samples, n_periods))

        # return the scaled values of s
        return s/s[:, [0]]
