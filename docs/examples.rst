More examples and recipes
==================================

This section goes through some examples.

Putting multiple plots on an figure:
##############################################

If you have a pandas `DataFrame` with columns "group", "T", and "C", then something like the following would work:

.. code-block:: python
    
    from lifelines import KaplanMeierFitter
    from matplotlib import pyplot as plt
    
    ax = plt.subplot(111)

    #group the data by column 'group'
    grouped_data = df.groupby("group")

    #iterate over the groups and plot the estimate + 95% intervals.
    unique_groups = grouped_data.groups.keys()
    unique_groups.sort()

    kmf = KaplanMeierFitter()
    for i, group in enumerate(unique_groups):
        data = grouped_data.get_group(group)
        kmf.fit(data["T"], data["C"], columns=[group])
        kmf.plot(ax=ax)
    

Example to set the index for a estimate 
##############################################

Suppose your dataset has lifetimes grouped near time 60, thus after fitting
`KaplanMeierFitter`, you survival function might look something like:

.. code-block:: python
    
    print kmf.survival_function_ 

        KM-estimate
    0          1.00
    47         0.99
    49         0.97
    50         0.96
    51         0.95
    52         0.91
    53         0.86
    54         0.84
    55         0.79
    56         0.74
    57         0.71
    58         0.67
    59         0.58
    60         0.49
    61         0.41
    62         0.31
    63         0.24
    64         0.19
    65         0.14
    66         0.10
    68         0.07
    69         0.04
    70         0.02
    71         0.01
    74         0.00


What you would really like is to have a predictable and full index from 40 to 75. (Notice that
in the above index, the last two points are seperated -- this is caused by observing no lifetimes
existing for at times 72 or 73) This is especially useful for comparing multiple survival functions at specific time points. To do this, all fitter methods accept a `timeline` argument: 

.. code-block:: python

    naf.fit( T, timeline=arange(40,75))
    print kmf.survival_function_ 

        KM-estimate
    40         1.00
    41         1.00
    42         1.00
    43         1.00
    44         1.00
    45         1.00
    46         1.00
    47         0.99
    48         0.99
    49         0.97
    50         0.96
    51         0.95
    52         0.91
    53         0.86
    54         0.84
    55         0.79
    56         0.74
    57         0.71
    58         0.67
    59         0.58
    60         0.49
    61         0.41
    62         0.31
    63         0.24
    64         0.19
    65         0.14
    66         0.10
    67         0.10
    68         0.07
    69         0.04
    70         0.02
    71         0.01
    72         0.01
    73         0.01
    74         0.00


lifelines will intelligently forward-fill the estimates to time points.

Example SQL query to get data from a table
##############################################

Below is a way to get an example dataset from a relation database (this may vary depending on your database schema):

.. code-block:: mysql

    SELECT 
      id, 
      DATEDIFF('dd', started_at, COALESCE(ended_at, CURRENT_DATE) ) AS "T", 
      (ended_at IS NULL) AS "C" 
    FROM some_tables

Explaination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each row is an `id`, a duration, and a boolean indicating whether the event occured or not. Recall that we denote a 
"True" if the event *did* occur, that is, `ended_at` is filled in (we observed the `ended_at`). Ex: 

==================   ============   ============
id                   T                      C
==================   ============   ============
10                   40                 True
11                   42                 False
12                   42                 False 
13                   36                 True
14                   33                 True
==================   ============   ============



