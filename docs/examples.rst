More examples and recipes
==================================


This section goes through some examples.

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
    

