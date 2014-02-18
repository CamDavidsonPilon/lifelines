More examples and recipes
==================================

This section goes through some examples.

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
---------------------------------------------

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



Putting multiple plots on an figure:
##############################################

If you have a pandas DataFrame with columns "group", "T", and "C", then something like the following would work:

.. code-block:: python

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    #group the data by 
    grouped_data = df.groupby("group")

    #iterate over the groups and plot the estimate + 95% intervals.
    unique_groups = grouped_data.groups.keys()
    unique_groups.sort()
    for i, group in enumerate(unique_groups):
        data = grouped_data.get_group(group)
        kmf.fit(data["T"], data["C"], columns=[group])
        kmf.plot(ax=ax)
    

