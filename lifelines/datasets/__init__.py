# -*- coding: utf-8 -*-
import pandas as pd
from pkg_resources import resource_filename


def load_dataset(filename, **kwargs):
    '''
    Load a dataset from lifelines.datasets

    Parameters:
    filename : for example "larynx.csv"
    usecols : list of columns in file to use

    Returns : Pandas dataframe
    '''
    return pd.read_csv(resource_filename('lifelines',
                                         'datasets/' + filename),
                       **kwargs)


def load_canadian_senators(**kwargs):
    return load_dataset('canadian_senators.csv', **kwargs)


def load_dd(**kwargs):
    """
    Classification of political regimes as democracy and dictatorship. Classification of democracies as parliamentary, semi-presidential (mixed) and presidential. Classification of dictatorships as military, civilian and royal. Coverage: 202 countries, from 1946 or year of independence to 2008.

    Cheibub, José Antonio, Jennifer Gandhi, and James Raymond Vreeland. 2010. “Democracy and Dictatorship Revisited.” Public Choice, vol. 143, no. 2-1, pp. 67-101.
    """
    return load_dataset('dd.csv', **kwargs)


def load_kidney_transplant(**kwargs):
    return load_dataset('kidney_transplant.csv', **kwargs)


def load_larynx(**kwargs):
    return load_dataset('larynx.csv', **kwargs)


def load_lung(**kwargs):
    return load_dataset('lung.csv', **kwargs)


def load_panel_test(**kwargs):
    return load_dataset('panel_test.csv', **kwargs)


def load_psychiatric_patients(**kwargs):
    return load_dataset('psychiatric_patients.csv', **kwargs)


def load_static_test(**kwargs):
    return load_dataset('static_test.csv', **kwargs)


def load_lcd(**kwargs):
    return load_dataset('CuZn-LeftCensoredDataset.csv', **kwargs)


def load_waltons(**kwargs):
    """
    Genotypes and number of days survived in Drosophila . Since we work with flies, we don't need to worry about left-censoring. We know the birth date of all flies. We do have issues with accidentally killing some or if some escape. These would be right-censored as we do not actually observe their death due to "natural" causes.
    """
    return load_dataset('waltons_dataset.csv', **kwargs)


def load_rossi(**kwargs):
    """
    This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.

    Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime:     Some Experimental Results. New York: Academic Press.     John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.
    """
    return load_dataset('rossi.csv', **kwargs)


def load_regression_dataset(**kwargs):
    return load_dataset('regression.csv', **kwargs)


def load_g3(**kwargs):
    return load_dataset('g3.csv', index_col=[0], **kwargs)
