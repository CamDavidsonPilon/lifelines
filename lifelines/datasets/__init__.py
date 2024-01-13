# -*- coding: utf-8 -*-
import pandas as pd
from importlib import resources


def _load_dataset(filename, **kwargs):
    """
    Load a dataset from lifelines.datasets

    Parameters
    ----------
    filename : string
        for example "larynx.csv"
    usecols : list
        list of columns in file to use

    Returns
    -------
        output: DataFrame
    """
    return pd.read_csv(resources.files("lifelines") / "datasets" / filename, engine="python", **kwargs)


def load_recur(**kwargs):
    """
    From ftp://ftp.wiley.com/public/sci_tech_med/survival/, first published in
    "Applied Survival Analysis: Regression Modeling of Time to Event Data, Second Edition"::

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
    return _load_dataset("recur.csv", **kwargs)


def load_multicenter_aids_cohort_study(**kwargs):
    """
    Originally in [1]::

        Size: (78, 4)

        AIDSY: date of AIDS diagnosis
        W: years from AIDS diagnosis to study entry
        T: years from AIDS diagnosis to minimum of death or censoring
        D: indicator of death during follow up


        i   AIDSY       W      T        D
        1   1990.425    4.575   7.575   0
        2   1991.250    3.750   6.750   0
        3   1992.014    2.986   5.986   0
        4   1992.030    2.970   5.970   0
        5   1992.072    2.928   5.928   0
        6   1992.220    2.780   4.688   1

    References
    ----------
    [1] Cole SR, Hudgens MG. Survival analysis in infectious disease research: describing events in time. AIDS. 2010;24(16):2423-31.
    """
    return _load_dataset("multicenter_aids_cohort.tsv", sep="\t", index_col=0, **kwargs)


def load_holly_molly_polly(**kwargs):
    """
    From https://stat.ethz.ch/education/semesters/ss2011/seminar/contents/presentation_10.pdf
    Used as a toy example for CoxPH in recurrent SA.::

          ID  Status  Stratum  Start(days)  Stop(days)  tx    T
        0  M       1        1            0         100   1  100
        1  M       1        2          100         105   1    5
        2  H       1        1            0          30   0   30
        3  H       1        2           30          50   0   20
        4  P       1        1            0          20   0   20

    """
    return _load_dataset("holly_molly_polly.tsv", sep=r"\s", **kwargs)


def load_leukemia(**kwargs):
    """
    Leukemia dataset.::

        Size: (42,5)
        Example:
                t  status  sex  logWBC  Rx
            0  35       0    1    1.45   0
            1  34       0    1    1.47   0
            2  32       0    1    2.20   0
            3  32       0    1    2.53   0
            4  25       0    1    1.78   0

    References
    ----------
    From http://web1.sph.emory.edu/dkleinb/allDatasets/surv2datasets/anderson.dat
    """
    return _load_dataset("anderson.csv", sep=" ", **kwargs)


def load_canadian_senators(**kwargs):
    """
    A history of Canadian senators in office.::

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
    return _load_dataset("canadian_senators.csv", **kwargs)


def load_dd(**kwargs):
    """
    Classification of political regimes as democracy and dictatorship.
    Classification of democracies as parliamentary, semi-presidential (mixed) and presidential.
    Classification of dictatorships as military, civilian and royal.
    Coverage: 202 countries, from 1946 or year of independence to 2008.::

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

    References
    ----------
    Cheibub, José Antonio, Jennifer Gandhi, and James Raymond Vreeland. 2010. “Democracy and Dictatorship Revisited.” Public Choice, vol. 143, no. 2-1, pp. 67-101.

    """
    return _load_dataset("dd.csv", **kwargs)


def load_kidney_transplant(**kwargs):
    """
    D.3 from Klein and Moeschberger Statistics for Biology and Health, 1997.

    ::

        Size: (863,6)
        Example:
            time             5
            death            0
            age             51
            black_male       0
            white_male       1
            black_female     0

    """
    return _load_dataset("kidney_transplant.csv", **kwargs)


def load_larynx(**kwargs):
    """
    ::

        Size: (89,6)
        Example:
            time  age  death  Stage_II  Stage_III  Stage_IV
             0.6   77      1         0          0         0
             1.3   53      1         0          0         0
             2.4   45      1         0          0         0
             2.5   57      0         0          0         0
             3.2   58      1         0          0         0

    """
    return _load_dataset("larynx.csv", **kwargs)


def load_lung(**kwargs):
    """
    Survival in patients with advanced lung cancer from the North Central Cancer Treatment Group. Performance scores rate how well the patient can perform usual daily activities.


    ::
        Size: (288,10)
        Example:
         inst  time  status  age  sex  ph.ecog  ph.karno  pat.karno  meal.cal  wt.loss
          3.0   306       1   74    1      1.0      90.0      100.0    1175.0      NaN
          3.0   455       1   68    1      0.0      90.0       90.0    1225.0     15.0
          3.0  1010       0   56    1      0.0      90.0       90.0       NaN     15.0
          5.0   210       1   57    1      1.0      90.0       60.0    1150.0     11.0
          1.0   883       1   60    1      0.0     100.0       90.0       NaN      0.0

    References
    -----------
    Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ. Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al. Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.

    """
    return _load_dataset("lung.csv", **kwargs)


def load_panel_test(**kwargs):
    """
    ::

        Size: (28,5)
        Example:
            id  t  E  var1  var2
             1  1  0   0.0     1
             1  2  0   0.0     1
             1  3  0   4.0     3
             1  4  1   8.0     4
             2  1  0   1.2     1

    """
    return _load_dataset("panel_test.csv", **kwargs)


def load_psychiatric_patients(**kwargs):
    """
    ::

        Size: (26,4)
        Example:
            Age   T  C  sex
             51   1  1    2
             58   1  1    2
             55   2  1    2
             28  22  1    2
             21  30  0    1

    """
    return _load_dataset("psychiatric_patients.csv", **kwargs)


def load_static_test(**kwargs):
    """
    ::

        Size: (7,5)
        Example:
            id  t  E  var1  var2
             1  4  1    -1    -1
             2  3  1    -2    -2
             3  3  0    -3    -3
             4  4  1    -4    -4
             5  2  1    -5    -5
             6  0  1    -6    -6
             7  2  1    -7    -7
    """
    return _load_dataset("static_test.csv", **kwargs)


def load_lcd(**kwargs):
    """
    Copper concentrations (µg/L) in shallow groundwater samples from two different geological zones in the San Joaquin Valley, California. The alluvial fan data include four
    different detection limits and the basin trough data include five different detection limits.

    Reference
    -----------
    Millard, S.P. and Deverel, S.J. (1988). Nonparametric statistical methods for comparing two sites based on data with multiple non-detect limits. Water Resources Research 24: doi: 10.1029/88WR03412. issn: 0043-1397.

    ::

        Size: (104,3)
        Example:
            C  T         group
            0  1  alluvial_fan
            0  1  alluvial_fan
            0  1  alluvial_fan
            0  1  alluvial_fan
            1  1  alluvial_fan
    """
    return _load_dataset("CuZn-LeftCensoredDataset.csv", **kwargs)


def load_nh4(**kwargs):
    """
    Ammonium (NH4) concentration (mg/L) in precipitation measured at Olympic National Park, Hoh Ranger Station (WA14), weekly or every other week from January 6, 2009 through December 20, 2011.

    Reference
    -----------
    National Atmospheric Deposition Program, National Trends Network (NADP/NTN).
    http://nadp.slh.wisc.edu/data/sites/siteDetails.aspx?net=NTN&id=WA14
    http://nadp.isws.illinois.edu/NTN/

    ::

        Size: (104,3)

    """
    return _load_dataset("nh4.csv", index_col=[0], **kwargs)


def load_waltons(**kwargs):
    """
    Genotypes and number of days survived in Drosophila. Since we work with flies, we don't need to worry about left-censoring. We know the birth date of all flies. We do have issues with accidentally killing some or if some escape. These would be right-censored as we do not actually observe their death due to "natural" causes.::

        Size: (163,3)
        Example:
             T  E    group
             6  1  miR-137
            13  1  miR-137
            13  1  miR-137
            13  1  miR-137
            19  1  miR-137

    """
    return _load_dataset("waltons_dataset.csv", **kwargs)


def load_rossi(**kwargs):
    """
    This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.::


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

    References
    ----------
    Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime:     Some Experimental Results. New York: Academic Press.     John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.

    """
    return _load_dataset("rossi.csv", **kwargs)


def load_regression_dataset(**kwargs):
    """
    Artificial regression dataset. Useful since there are no ties in this dataset.
    Slightly edit in v0.15.0 to achieve this, however.::

        Size: (200,5)
        Example:
                var1      var2      var3          T  E
            0.595170  1.143472  1.571079  14.785479  1
            0.209325  0.184677  0.356980   7.336734  1
            0.693919  0.071893  0.557960   5.271527  1
            0.443804  1.364646  0.374221  11.684168  1
            1.613324  0.125566  1.921325   7.637764  1

    """
    return _load_dataset("regression.csv", **kwargs)


def load_g3(**kwargs):
    """
    ::

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
    return _load_dataset("g3.csv", **kwargs)


def load_stanford_heart_transplants(**kwargs):
    """
    This is a classic dataset for survival regression with time
    varying covariates. The original dataset is from [1], and
    this dataset is from R's survival library.::

        Size: (172, 8)
        Example:
            start  stop  event        age      year  surgery  transplant  id
              0.0  50.0      1 -17.155373  0.123203        0           0   1
              0.0   6.0      1   3.835729  0.254620        0           0   2
              0.0   1.0      0   6.297057  0.265572        0           0   3
              1.0  16.0      1   6.297057  0.265572        0           1   3
              0.0  36.0      0  -7.737166  0.490075        0           0   4

    References
    ----------
    [1] J Crowley and M Hu. Covariance analysis of heart transplant survival data. J American
        Statistical Assoc, 72:27–36, 1977.

    """
    return _load_dataset("stanford_heart.csv", **kwargs)


def load_gbsg2(**kwargs):
    """
    A data frame containing the observations from the GBSG2 study of 686 women.::

        Size: (686,10)
        Example:
            horTh           yes
            age             56
            menostat        Post
            tsize           12
            tgrade          II
            pnodes          7
            progrec         61
            estrec          77
            time            2018
            cens            1

    References
    ----------
    W. Sauerbrei and P. Royston (1999). Building multivariable prognostic and diagnostic models: transformation of the predictors by using fractional polynomials. Journal of the Royal Statistics Society Series A, Volume 162(1), 71–94

    M. Schumacher, G. Basert, H. Bojar, K. Huebner, M. Olschewski, W. Sauerbrei, C. Schmoor, C. Beyerle, R.L.A. Neumann and H.F. Rauschecker for the German Breast Cancer Study Group (1994), Randomized 2 × 2 trial evaluating hormonal treatment and the duration of chemotherapy in node- positive breast cancer patients. Journal of Clinical Oncology, 12, 2086–2093
    """
    return _load_dataset("gbsg2.csv", **kwargs)


def load_dfcv():
    """
    A toy example of a time dependent dataset. ::


        Size: (14, 6)
        Example:

         start  group  z  stop  id  event
             0    1.0  0   3.0   1   True
             0    1.0  0   5.0   2  False
             0    1.0  1   5.0   3   True
             0    1.0  0   6.0   4   True


    References
    -----------
    From http://www.math.ucsd.edu/~rxu/math284/slect7.pdf
    """
    from lifelines.datasets.dfcv_dataset import dfcv

    return dfcv.copy()


def load_lymphoma(**kwargs):
    """
    ::

        Size: (80, 3)
        Example:

          Stage_group  Time  Censor
                    1     6       1
                    1    19       1
                    1    32       1
                    1    42       1
                    1    42       1

    References
    ----------
    From https://www.statsdirect.com/help/content/survival_analysis/logrank.htm
    """
    return _load_dataset("lymphoma.csv", **kwargs)


def load_diabetes(**kwargs):
    """
    An interval censored dataset.

    References
    ----------
    Borch-Johnsens, K, Andersen, P and Decker, T (1985). "The effect of proteinuria on relative mortality in Type I (insulin-dependent) diabetes mellitus." Diabetologia, 28, 590-596.

    ::

        Size: (731, 3)
        Example:

           left  right  gender
             24     27    male
             22     22  female
             37     39    male
             20     20    male
              1     16    male
              8     20  female
             14     14    male
    """

    return _load_dataset("interval_diabetes.csv", index_col=0, **kwargs)


def load_lupus(**kwargs):
    """
    See https://projecteuclid.org/download/pdf_1/euclid.aos/1176345693

    Note
    ------
    I transcribed this from the original paper, and highly suspect there are differences. See Notes below.

    References
    -----------
    Merrell, M., & Shulman, L. E. (1955). Determination of prognosis in chronic disease, illustrated by systemic lupus erythematosus. Journal of Chronic Diseases, 1(1), 12–32. doi:10.1016/0021-9681(55)90018-7


    Notes
    ------

    In lifelines v0.23.7, two rows were updated with more correct data (transcription problems originally.)

    """
    return _load_dataset("merrell1955.csv", index_col=0, **kwargs)


def load_lymph_node(**kwargs):
    """
    References
    -----------
    Schmoor, C., Sauerbrei, W. Bastert, G., Schumacher, M. (2000). Role of Isolated Locoregional Recurrence of Breast Cancer: Results of Four Prospective Studies. Journal of Clinical Oncology, 18(8), 1696-1708.

    Schumacher, M., Bastert, G., Bojar, H., Hiibner, K., Olschewski, M., Sauerbrei, W., Schmoor, C., Beyerle, C., Neumann, R.L.A. and Rauschecker, H.F. for the German Breast Cancer Study Group (GBSG) (1994). A randomized 2 x 2 trial evaluating hormonal treatment and the duration of chemotherapy in node-positive breast cancer patients. Journal of Clinical Oncology, 12, 2086-2093.

    Hosmer, D.W. and Lemeshow, S. and May, S. (2008). Applied Survival Analysis: Regression Modeling of Time to Event Data: Second Edition, John Wiley and Sons Inc., New York, NY
    """
    return _load_dataset("lymph_node.csv", index_col=0, **kwargs)


def load_c_botulinum_lag_phase(**kwargs):
    """
    A dataset from [1] that represents the duration of the lag phase for C. botulinum, measured in days, at 30C. The data is left and right censored.
    Note that the table does not have 6% NaCl, but the authors mention no growth occurred (we can infer lag time > 85D then)

    References
    -----------
    Montville, THOMAS J. "Interaction of pH and NaCl on culture density of Clostridium botulinum 62A." Appl. Environ. Microbiol. 46.4 (1983): 961-963.

    """
    return _load_dataset("c_botulinum_lag_phase.csv", **kwargs)


def load_mice(**kwargs):
    """
    A dataset of interval-censored observations of mice tumors in two different environments.

    References
    -----------
    Hoel D. and Walburg, H.,(1972), Statistical analysis of survival experiments, The Annals of Statistics, 18, 1259-1294
    """
    return _load_dataset("mice.csv", **kwargs)
