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


def load_recur(**kwargs):
    """
    From ftp://ftp.wiley.com/public/sci_tech_med/survival/, first published in
    "Applied Survival Analysis: Regression Modeling of Time to Event Data, Second Edition"

    ID          Subject Identification        1 - 400
    AGE         Age                           years
    TREAT       Treatment Assignment          0 = New
                                              1 = Old
    TIME0       Day of Previous Episode       Days
    TIME1       Day of New Episode            Days
                  or censoring
    CENSOR      Indicator for Soreness        1 = Episode Occurred
                  Episode or Censoring            at TIME1
                                              0 = Censored
    EVENT       Soreness Episode Number       0 to at most 4

    Size: (1296, 7)
    Example:
        ID,AGE,TREAT,TIME0,TIME1,CENSOR,EVENT
        1,43,0,9,56,1,3
        1,43,0,56,88,1,4
        1,43,0,0,6,1,1
        1,43,0,6,9,1,2

    """
    return load_dataset('recur.csv', **kwargs)


def load_holly_molly_polly(**kwargs):
    """
    From https://stat.ethz.ch/education/semesters/ss2011/seminar/contents/presentation_10.pdf
    Used as a toy example for CoxPH in recurrent SA.

     ID Status Stratum Start(days) Stop(days) tx T
     M 1 1 0 100 1 100
     M 1 2 100 105 1 5
     H 1 1 0 30 0 30
     H 1 2 30 50 0 20
     P 1 1 0 20 0 20
     P 1 2 20 60 0 40
     P 1 3 60 85 0 25

    """
    return load_dataset('holly_molly_polly.tsv', sep="\s", **kwargs)


def load_leukemia(**kwargs):
    """
    Leukemia dataset. From http://web1.sph.emory.edu/dkleinb/allDatasets/surv2datasets/anderson.dat
    Size: (42,5)
    Example:
            t  status  sex  logWBC  Rx
        0  35       0    1    1.45   0
        1  34       0    1    1.47   0
        2  32       0    1    2.20   0
        3  32       0    1    2.53   0
        4  25       0    1    1.78   0
    """
    return load_dataset('anderson.csv', sep=" ", **kwargs)


def load_canadian_senators(**kwargs):
    """
    A history of Canadian senators in office.

    Size: (933,10)
    Example:
        Name                                        Abbott, John Joseph Caldwell
        Political Affiliation at Appointment                Liberal-Conservative
        Province / Territory                                              Quebec
        Appointed on the advice of                     Macdonald, John Alexander
        Term (yyyy.mm.dd)                       1887.05.12 - 1893.10.30  (Death)
        start_date                                           1887-05-12 00:00:00
        end_date                                             1893-10-30 00:00:00
        reason                                                             Death
        diff_days                                                           2363
        observed                                                            True
    """
    return load_dataset('canadian_senators.csv', **kwargs)


def load_dd(**kwargs):
    """
    Classification of political regimes as democracy and dictatorship. Classification of democracies as parliamentary, semi-presidential (mixed) and presidential. Classification of dictatorships as military, civilian and royal. Coverage: 202 countries, from 1946 or year of independence to 2008.

    Cheibub, José Antonio, Jennifer Gandhi, and James Raymond Vreeland. 2010. “Democracy and Dictatorship Revisited.” Public Choice, vol. 143, no. 2-1, pp. 67-101.

    Size: (1808, 12)
    Example:
        ctryname                                                   Afghanistan
        cowcode2                                                           700
        politycode                                                         700
        un_region_name                                           Southern Asia
        un_continent_name                                                 Asia
        ehead                                              Mohammad Zahir Shah
        leaderspellreg       Mohammad Zahir Shah.Afghanistan.1946.1952.Mona...
        democracy                                                Non-democracy
        regime                                                        Monarchy
        start_year                                                        1946
        duration                                                             7
        observed                                                             1
    """
    return load_dataset('dd.csv', **kwargs)


def load_kidney_transplant(**kwargs):
    """
    Size: (863,6)
    Example:
        time             5
        death            0
        age             51
        black_male       0
        white_male       1
        black_female     0

    """
    return load_dataset('kidney_transplant.csv', **kwargs)


def load_larynx(**kwargs):
    """
    Size: (89,6)
    Example:
            time  age  death  Stage II  Stage III  Stage IV
        0    0.6   77      1         0          0         0
        1    1.3   53      1         0          0         0
        2    2.4   45      1         0          0         0
        3    2.5   57      0         0          0         0
        4    3.2   58      1         0          0         0

    """
    return load_dataset('larynx.csv', **kwargs)


def load_lung(**kwargs):
    """
    Size: (288,10)
    Example:
        inst            3
        time          306
        status          2
        age            74
        sex             1
        ph.ecog         1
        ph.karno       90
        pat.karno     100
        meal.cal     1175
        wt.loss       NaN

    """
    return load_dataset('lung.csv', **kwargs)


def load_panel_test(**kwargs):
    """
    Size: (28,5)
    Example:
           id  t  E  var1  var2
        0   1  1  0   0.0     1
        1   1  2  0   0.0     1
        2   1  3  0   4.0     3
        3   1  4  1   8.0     4
        4   2  1  0   1.2     1

    """
    return load_dataset('panel_test.csv', **kwargs)


def load_psychiatric_patients(**kwargs):
    """
    Size: (26,4)
    Example:
           Age   T  C  sex
        0   51   1  1    2
        1   58   1  1    2
        2   55   2  1    2
        3   28  22  1    2
        4   21  30  0    1

    """
    return load_dataset('psychiatric_patients.csv', **kwargs)


def load_static_test(**kwargs):
    """
    Size: (7,5)
    Example:
           id  t  E  var1  var2
        0   1  4  1    -1    -1
        1   2  3  1    -2    -2
        2   3  3  0    -3    -3
        3   4  4  1    -4    -4
        4   5  2  1    -5    -5
        5   6  0  1    -6    -6
        6   7  2  1    -7    -7
    """
    return load_dataset('static_test.csv', **kwargs)


def load_lcd(**kwargs):
    """
    Size: (104,3)
    Example:
           C  T         group
        0  0  1  alluvial_fan
        1  0  1  alluvial_fan
        2  0  1  alluvial_fan
        3  0  1  alluvial_fan
        4  1  1  alluvial_fan
    """
    return load_dataset('CuZn-LeftCensoredDataset.csv', **kwargs)


def load_waltons(**kwargs):
    """
    Genotypes and number of days survived in Drosophila . Since we work with flies, we don't need to worry about left-censoring. We know the birth date of all flies. We do have issues with accidentally killing some or if some escape. These would be right-censored as we do not actually observe their death due to "natural" causes.

    Size: (163,3)
    Example:
            T  E    group
        0   6  1  miR-137
        1  13  1  miR-137
        2  13  1  miR-137
        3  13  1  miR-137
        4  19  1  miR-137
    """
    return load_dataset('waltons_dataset.csv', **kwargs)


def load_rossi(**kwargs):
    """
    This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.

    Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime:     Some Experimental Results. New York: Academic Press.     John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.

    Size: (432,9)
    Example:
        week      20
        arrest     1
        fin        0
        age       27
        race       1
        wexp       0
        mar        0
        paro       1
        prio       3
    """
    return load_dataset('rossi.csv', **kwargs)


def load_regression_dataset(**kwargs):
    """
    Artificial regression dataset

    Size: (200,5)
    Example:
               var1      var2      var3          T  E
        0  0.595170  1.143472  1.571079  14.785479  1
        1  0.209325  0.184677  0.356980   7.336734  1
        2  0.693919  0.071893  0.557960   5.271527  1
        3  0.443804  1.364646  0.374221  11.684168  1
        4  1.613324  0.125566  1.921325   7.637764  1
    """
    return load_dataset('regression.csv', **kwargs)


def load_g3(**kwargs):
    """

    Size: (17,7)
    Example:
        no.               1
        age              41
        sex          Female
        histology    Grade3
        group           RIT
        event          True
        time             53
    """
    return load_dataset('g3.csv', **kwargs)
