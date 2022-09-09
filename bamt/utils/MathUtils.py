import numpy as np
import pandas as pd
import math
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture
from scipy.stats.distributions import chi2
from sklearn.preprocessing import OrdinalEncoder


def lrts_comp(data):
    n = 0
    biggets_p = -1 * np.infty
    comp_biggest = 0
    max_comp = 10
    if len(data) < max_comp:
        max_comp = len(data)
    for i in range(1, max_comp + 1, 1):
        gm1 = GaussianMixture(n_components=i, random_state=0)
        gm2 = GaussianMixture(n_components=i + 1, random_state=0)
        gm1.fit(data)
        ll1 = np.mean(gm1.score_samples(data))
        gm2.fit(data)
        ll2 = np.mean(gm2.score_samples(data))
        LR = 2 * (ll2 - ll1)
        p = chi2.sf(LR, 1)
        if p > biggets_p:
            biggets_p = p
            comp_biggest = i
        n = comp_biggest
    return n


def mix_norm_cdf(x, weights, means, covars):
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * \
            stats.norm.cdf(x, loc=means[i][0], scale=covars[i][0][0])
    return mcdf


def theoretical_quantile(data, n_comp):
    model = GaussianMixture(n_components=n_comp, random_state=0)
    model.fit(data)
    q = []
    x = []
    # step =  ((np.max(model.sample(100000)[0])) - (np.min(model.sample(100000)[0])))/1000
    step = (np.max(data) - np.min(data)) / 1000
    d = np.arange(np.min(data), np.max(data), step)
    for i in d:
        x.append(i)
        q.append(mix_norm_cdf(i, model.weights_,
                 model.means_, model.covariances_))
    return x, q


def quantile_mix(p, vals, q):
    ind = q.index(min(q, key=lambda x: abs(x - p)))
    return vals[ind]


def probability_mix(val, vals, q):
    ind = vals.index(min(vals, key=lambda x: abs(x - val)))
    return (q[ind])


def sum_dist(data, vals, q):
    percs = np.linspace(1, 100, 10)
    x = np.quantile(data, percs / 100)
    y = []
    for p in percs:
        y.append(quantile_mix(p / 100, vals, q))
    dist = 0
    for xi, yi in zip(x, y):
        dist = dist + (abs(-1 * xi + yi)) / math.sqrt(2)
    return dist


def component(data, columns, method):
    n = 1
    max_comp = 10
    x = []
    if data.shape[0] < max_comp:
        max_comp = data.shape[0]
    if len(columns) == 1:
        x = np.transpose([data[columns[0]].values])
    else:
        x = data[columns].values
    if method == 'aic':
        lowest_aic = np.infty
        comp_lowest = 0
        for i in range(1, max_comp + 1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            aic1 = gm1.aic(x)
            if aic1 < lowest_aic:
                lowest_aic = aic1
                comp_lowest = i
            n = comp_lowest

    if method == 'bic':
        lowest_bic = np.infty
        comp_lowest = 0
        for i in range(1, max_comp + 1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            bic1 = gm1.bic(x)
            if bic1 < lowest_bic:
                lowest_bic = bic1
                comp_lowest = i
            n = comp_lowest

    if method == 'LRTS':
        n = lrts_comp(x)
    if method == 'quantile':
        biggest_p = -1 * np.infty
        comp_biggest = 0
        for i in range(1, max_comp, 1):
            vals, q = theoretical_quantile(x, i)
            dist = sum_dist(x, vals, q)
            p = probability_mix(dist, vals, q)
            if p > biggest_p:
                biggest_p = p
                comp_biggest = i
        n = comp_biggest
    return n


def get_n_nearest(data, columns, corr=False, number_close=5):
    """Returns N nearest neighbors for every column of dataframe, added into list

    Args:
        data (DataFrame): Proximity matrix
        columns (list): df.columns.tolist()
        corr (bool, optional): _description_. Defaults to False.
        number_close (int, optional): Number of nearest neighbors. Defaults to 5.

    Returns:
        groups
    """
    groups = []
    for c in columns:
        if corr:
            close_ind = data[c].sort_values(ascending=False).index.tolist()
        else:
            close_ind = data[c].sort_values().index.tolist()
        groups.append(close_ind[0:number_close + 1])

    return groups


def get_proximity_matrix(df, proximity_metric) -> pd.DataFrame:
    """Returns matrix of proximity matrix of the dataframe, dataframe must be coded first if it contains
                                                                                                    categorical data

    Args:
        df (DataFrame): data
        df_coded (DataFrame): same data, but coded
        proximity_metric (str): 'MI' or 'corr'

    Returns:
        df_distance: mutual information matrix
    """

    encoder = OrdinalEncoder()
    df_coded = df
    columnsToEncode = list(df_coded.select_dtypes(
        include=['category', 'object']))

    df_coded[columnsToEncode] = encoder.fit_transform(
        df_coded[columnsToEncode])

    df_distance = pd.DataFrame(data=np.zeros(
        (len(df.columns), len(df.columns))), columns=df.columns)
    df_distance.index = df.columns

    if proximity_metric == 'MI':
        for c1 in df.columns:
            for c2 in df.columns:
                dist = mutual_info_score(
                    df_coded[c1].values, df_coded[c2].values)
                df_distance.loc[c1, c2] = dist

    elif proximity_metric == 'corr':
        df_distance = df_coded.corr(method='pearson')

    return df_distance


def get_brave_matrix(df_columns, proximity_matrix, n_nearest=5) -> pd.DataFrame:
    """Returns matrix Brave coeffitients of the DataFrame, requires proximity measure to be calculated

    Args:
        df_columns (DataFrame): data.columns
        proximity_matrix (DataFrame): may be generated by get_mutual_info_score_matrix() function or
                                                                                                correlation from scipy
        n_nearest (int, optional): _description_. Defaults to 5.

    Returns:
        brave_matrix: DataFrame of Brave coefficients
    """

    brave_matrix = pd.DataFrame(data=np.zeros(
        (len(df_columns), len(df_columns))), columns=df_columns)
    brave_matrix.index = df_columns

    groups = get_n_nearest(proximity_matrix, df_columns.tolist(),
                           corr=True, number_close=n_nearest)

    counter_zeroer = .0

    for c1 in df_columns:
        for c2 in df_columns:
            a = counter_zeroer
            b = counter_zeroer
            c = counter_zeroer
            d = counter_zeroer
            if c1 != c2:
                for g in groups:
                    if (c1 in g) & (c2 in g):
                        a += 1
                    if (c1 in g) & (c2 not in g):
                        b += 1
                    if (c1 not in g) & (c2 in g):
                        c += 1
                    if (c1 not in g) & (c2 not in g):
                        d += 1

                if (a + c) * (b + d) != 0 and (a + b) * (c + d) != 0:

                    br = (a * len(groups) + (a + c) * (a + b)) / ((math.sqrt((a + c) *
                                                                             (b + d))) * (math.sqrt((a + b) * (c + d))))
                else:
                    br = (a * len(groups) + (a + c) * (a + b)) / 0.0000000001
                brave_matrix.loc[c1, c2] = br

    return brave_matrix
