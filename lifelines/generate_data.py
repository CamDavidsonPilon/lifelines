from __future__ import print_function, division
import numpy as np
import pandas as pd


def make_raw_article_data():
    """
    A function to generate the customer level data used in [1]

    :return: pandas DataFrame
    """

    # List of number of customers lost per year per quality level
    highend_lost = np.asarray([0, 131, 257, 347, 407, 449, 483, 509])
    regular_lost = np.asarray([0, 369, 532, 618, 674, 711, 738, 759])

    # high-end
    data_he = np.zeros((1000, 5), dtype=float)

    # Start age column with max-age, we will change this later
    data_he[:, -2] = 8

    # is high end indicator
    data_he[:, 1] = 1

    for age in reversed(highend_lost):
        data_he[:age, -2] -= 1

    # regulars
    data_re = np.zeros((1000, 5), dtype=float)
    data_re[:, -2] = 8

    for age in reversed(regular_lost):
        data_re[:age, -2] -= 1

    # those with age 8 are still alive (here we assume they all belong to the
    # same cohort)
    data_he[:, -1][data_he[:, -2] == 8] = 1
    data_re[:, -1][data_re[:, -2] == 8] = 1

    out_data = np.concatenate((data_he, data_re), axis=0)
    np.random.shuffle(out_data)

    # ids
    out_data[:, 0] = np.arange(out_data.shape[0]) + 1000

    # random field
    out_data[:, 2] = np.random.randn(out_data.shape[0])

    data = pd.DataFrame(data=out_data, columns=['id', 'is_high_end', 'random', 'age', 'alive'])

    return data

# Define parameters of the model!
params = dict(alpha=dict(bias=0.2,
                         categ={'cat_a': 0.09736,
                                'cat_b': -0.368,
                                'cat_c': -1e-2},
                         count=0.065,
                         numer=0.292),
              beta=dict(bias=0.87,
                        categ={'cat_a': -0.12,
                               'cat_b': -0.4425,
                               'cat_c': 0.69},
                        count=-0.148,
                        numer=0.021))


def get_age(alpha, beta, max_age=10):
    """
    A function to simulate the life of a sample given its alpha and beta
    parameters

    :param alpha:
    :param beta:
    :param max_age:
    :return:
    """
    age = 1
    alive = 1

    pchurn = np.random.beta(alpha, beta)

    while age < max_age:
        if np.random.random() <= pchurn:
            alive = 0
            break

        age += 1

    return age, alive

def compute_alpha(row):

    # Get alpha params
    pdict = params['alpha']

    # Start with bias
    alpha = np.exp(pdict['bias'])

    # add categorical contribution
    alpha *= np.exp(pdict['categ'][row['category']])

    # Add count and numerical contributions
    alpha *= np.exp(pdict['count'] * row['counts'] +
                    pdict['numer'] * row['numerical'])
    # add noise
    alpha *= np.exp(2e-2 * np.random.randn())
    return alpha

def compute_beta(row):

    # Get beta params
    pdict = params['beta']

    # Start with bias
    beta = np.exp(pdict['bias'])

    # add categorical contribution
    beta *= np.exp(pdict['categ'][row['category']])

    # Add count and numerical contributions
    beta *= np.exp(pdict['count'] * row['counts'] +
                   pdict['numer'] * row['numerical'])
    # add noise
    beta *= np.exp(1e-2 * np.random.randn())

    return beta

def compute_age(row):
    max_age = np.random.choice([8, 9, 10, 11], 1, p=[0.3, 0.3, 0.2, 0.2])[0]
    age, alive = get_age(row['alpha_true'], row['beta_true'], max_age=max_age)
    return age, alive


def simulate_data(size=10000, max_age=10):

    data = pd.DataFrame()
    data['id'] = np.arange(size)

    # add categories
    data['category'] = np.random.choice(['cat_a', 'cat_b', 'cat_c'],
                                        size,
                                        p=[0.47, 0.36, 0.17])

    # transform cotegory type
    data['category'] = data['category'].astype('category')

    # Add counts feature
    data['counts'] = np.random.poisson(lam=0.25, size=size)

    # add numerical gaussian feature
    data['numerical'] = 0.5 * np.random.randn(size) + 1

    # Add true alpha and beta params
    data['alpha_true'] = data.apply(compute_alpha, axis=1)
    data['beta_true'] = data.apply(compute_beta, axis=1)

    # Simulate age
    sim = data.apply(compute_age, axis=1)
        
    # Update age values    
    data['age'] = [s[0] for s in sim]

    # for simplicity we assume all come from same cohort, so it is easy to set
    # alive value
    data['alive'] = [s[1] for s in sim]

    # split in half
    tr = data.iloc[:size//2]
    te = data.iloc[size//2:].reset_index().drop('index', axis=1)

    return {'train': tr, 'test': te, 'params': params}


if __name__ == '__main__':
    print(make_raw_article_data().head())
    print(simulate_data(100))
