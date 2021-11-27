from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from concurrent.futures import ThreadPoolExecutor
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
import itertools
from sys import float_info
from Utils.MathUtils import *
from gmr import GMM
import random


class BaseNode(object):
    def __init__(self, name):
        self.name = name
        self.type = 'abstract'

        self.disc_parents = None
        self.cont_parents = None
        self.children = None

    def __repr__(self):
        return f"{self.name}"


class DiscreteNode(BaseNode):
    def __init__(self, name):
        super(DiscreteNode, self).__init__(name)
        self.type = 'Discrete'

    def fit_parameters(self, data):
        def worker(node):
            parents = node.disc_parents + node.cont_parents
            if not parents:
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                cprob = list(dict(sorted(dist.items())).values())
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            else:
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[parents + [node.name]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
            return {"cprob": cprob, 'vals': vals}

        pool = ThreadPoolExecutor(3)
        future = pool.submit(worker, self)
        return future.result()

    def choose(self, node_info, pvals=None):
        random.seed()
        vals = node_info['vals']
        if not pvals:
            dist = node_info['cprob']
        else:
            dist = node_info['cprob'][str(pvals)]
        lbound = 0
        ubound = 0
        rand = random.random()
        for interval in range(len(dist)):
            ubound += dist[interval]
            if lbound <= rand < ubound:
                rindex = interval
                break
            else:
                lbound = ubound

        return vals[rindex]


class GaussianNode(BaseNode):
    def __init__(self, name):
        super(GaussianNode, self).__init__(name)
        self.type = 'Gaussian'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.cont_parents
        if parents:
            model = linear_model.LinearRegression()
            predict = []
            if len(parents) == 1:
                model.fit(np.transpose([data[parents[0]].values]), data[self.name].values)
                predict = model.predict(np.transpose([data[parents[0]].values]))
            else:
                model.fit(data[parents].values, data[self.name].values)
                predict = model.predict(data[parents].values)
            variance = mse(data[self.name].values, predict)
            return {"mean_base": model.intercept_,
                    "mean_scal": list(model.coef_),
                    "variance": variance}
        else:
            mean_base = np.mean(data[self.name].values)
            variance = np.var(data[self.name].values)
            return {"mean_base": mean_base,
                    "mean_scal": [],
                    "variance": variance}


class ConditionalGaussianNode(BaseNode):
    def __init__(self, name):
        super(ConditionalGaussianNode, self).__init__(name)
        self.type = 'ConditionalGaussian'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.cont_parents
        if self.disc_parents and self.cont_parents:
            hycprob = dict()
            values = []
            combinations = []
            for d_p in self.disc_parents:
                values.append(np.unique(data[d_p].values))
            for xs in itertools.product(*values):
                combinations.append(list(xs))
            for comb in combinations:
                mask = np.full(len(data), True)
                for col, val in zip(self.disc_parents, comb):
                    mask = (mask) & (data[col] == val)
                new_data = data[mask]
                mean_base = np.nan
                variance = np.nan
                predict = []
                if new_data.shape[0] != 0:
                    model = linear_model.LinearRegression()
                    if len(self.cont_parents) == 1:
                        model.fit(np.transpose([new_data[self.cont_parents[0]].values]), new_data[self.name].values)
                        predict = model.predict(np.transpose([new_data[self.cont_parents[0]].values]))
                    else:
                        model.fit(new_data[self.cont_parents].values, new_data[self.name].values)
                        predict = model.predict(new_data[self.cont_parents].values)
                    key_comb = [str(x) for x in comb]
                    mean_base = model.intercept_
                    variance = mse(new_data[self.name].values, predict)
                    hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base,
                                              'mean_scal': list(model.coef_)}
                else:
                    scal = list(np.full(len(self.cont_parents), np.nan))
                    key_comb = [str(x) for x in comb]
                    hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
            return {"hybcprob": hycprob}
            # Conditional Gaussian Node
        if self.disc_parents and not self.cont_parents:
            hycprob = dict()
            values = []
            combinations = []
            for d_p in self.disc_parents:
                values.append(np.unique(data[d_p].values))
            for xs in itertools.product(*values):
                combinations.append(list(xs))
            for comb in combinations:
                mask = np.full(len(data), True)
                for col, val in zip(self.disc_parents, comb):
                    mask = (mask) & (data[col] == val)
                new_data = data[mask]
                if new_data.shape[0] != 0:
                    mean_base = np.mean(new_data[self.name].values)
                    variance = np.var(new_data[self.name].values)
                    key_comb = [str(x) for x in comb]
                    hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                else:
                    mean_base = np.nan
                    variance = np.nan
                    key_comb = [str(x) for x in comb]
                    hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
            return {"hybcprob": hycprob}


class MixtureGaussianNode(BaseNode):
    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = 'MixtureGaussian'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.disc_parents
        if not parents:
            n_comp = int((component(data, [self.name], 'aic') + component(data, [self.name],
                                                                          'bic')) / 2)  # component(data, [node], 'LRTS')#
            # n_comp = 3
            gmm = GMM(n_components=n_comp)
            gmm.from_samples(np.transpose([data[self.name].values]))
            means = gmm.means.tolist()
            cov = gmm.covariances.tolist()
            # weigts = np.transpose(gmm.to_responsibilities(np.transpose([data[node].values])))
            w = gmm.priors.tolist()  # []
            # for row in weigts:
            #     w.append(np.mean(row))
            # if self.children:
            #     node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
            #                                 "variance": cov, "type": "lg", "children": children}
            # else:
            #     node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
            #                                 "variance": cov, "type": "lg", "children": None}
            return {"mean_base": means, "mean_scal": w, "parents": parents,
                    "variance": cov, "type": "lg", "children": self.children}
        if parents:
            # Mixture Gaussian Node
            if not self.disc_parents and self.cont_parents:
                nodes = [self.name] + self.cont_parents
                new_data = data[nodes][:]
                new_data.reset_index(inplace=True, drop=True)
                n_comp = int((component(new_data, nodes, 'aic') + component(new_data, nodes,
                                                                            'bic')) / 2)  # component(new_data, nodes, 'LRTS')#
                # n_comp = 3
                gmm = GMM(n_components=n_comp)
                gmm.from_samples(new_data[nodes].values)
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                return {"mean_base": means,
                        "mean_scal": w,
                        "variance": cov}


class ConditionalMixtureGaussianNode(BaseNode):
    def __init__(self, name):
        super(ConditionalMixtureGaussianNode, self).__init__(name)
        self.type = 'ConditionalMixtureGaussian'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.cont_parents
        if self.disc_parents and self.cont_parents:
            hycprob = dict()
            values = []
            combinations = []
            for d_p in self.disc_parents:
                values.append(np.unique(data[d_p].values))
            for xs in itertools.product(*values):
                combinations.append(list(xs))
            for comb in combinations:
                mask = np.full(len(data), True)
                for col, val in zip(self.disc_parents, comb):
                    mask = (mask) & (data[col] == val)
                new_data = data[mask]
                new_data.reset_index(inplace=True, drop=True)
                key_comb = [str(x) for x in comb]
                nodes = [self.name] + self.cont_parents
                if new_data.shape[0] > 5:
                    n_comp = int((component(new_data, nodes, 'aic') + component(new_data, nodes,
                                                                                'bic')) / 2)  # component(new_data, nodes, 'LRTS')#int((component(new_data, nodes, 'aic') + component(new_data, nodes, 'bic')) / 2)
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp)
                    gmm.from_samples(new_data[nodes].values)
                    means = gmm.means.tolist()
                    cov = gmm.covariances.tolist()
                    # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                    w = gmm.priors.tolist()  # []
                    # for row in weigts:
                    #     w.append(np.mean(row))
                    hycprob[str(key_comb)] = {'variance': cov, 'mean_base': means, 'mean_scal': w}
                else:
                    if new_data.shape[0] != 0:
                        n_comp = 1
                        gmm = GMM(n_components=n_comp)
                        gmm.from_samples(new_data[nodes].values)
                        means = gmm.means.tolist()
                        cov = gmm.covariances.tolist()
                        # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                        w = gmm.priors.tolist()
                        # for row in weigts:
                        #     w.append(np.mean(row))
                        hycprob[str(key_comb)] = {'variance': cov, 'mean_base': means, 'mean_scal': w}
                    else:
                        hycprob[str(key_comb)] = {'variance': np.nan, 'mean_base': np.nan, 'mean_scal': []}
            return {"hybcprob": hycprob}

        if self.disc_parents and not self.cont_parents:
            hycprob = dict()
            values = []
            combinations = []
            for d_p in self.disc_parents:
                values.append(np.unique(data[d_p].values))
            for xs in itertools.product(*values):
                combinations.append(list(xs))
            for comb in combinations:
                mask = np.full(len(data), True)
                for col, val in zip(self.disc_parents, comb):
                    mask = (mask) & (data[col] == val)
                new_data = data[mask]
                key_comb = [str(x) for x in comb]
                if new_data.shape[0] > 5:
                    n_comp = int((component(new_data, [self.name], 'aic') + component(new_data, [self.name],
                                                                                      'bic')) / 2)  # component(new_data, [node], 'LRTS')#int((component(new_data, [node], 'aic') + component(new_data, [node], 'bic')) / 2)
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp)
                    gmm.from_samples(np.transpose([new_data[self.name].values]))
                    means = gmm.means.tolist()
                    cov = gmm.covariances.tolist()
                    # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                    w = gmm.priors.tolist()  # []
                    # for row in weigts:
                    #     w.append(np.mean(row))
                    hycprob[str(key_comb)] = {'variance': cov, 'mean_base': means, 'mean_scal': w}
                else:
                    if new_data.shape[0] != 0:
                        n_comp = 1
                        gmm = GMM(n_components=n_comp)
                        gmm.from_samples(np.transpose([new_data[self.name].values]))
                        means = gmm.means.tolist()
                        cov = gmm.covariances.tolist()
                        # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                        w = gmm.priors.tolist()  # []
                        # for row in weigts:
                        #     w.append(np.mean(row))
                        hycprob[str(key_comb)] = {'variance': cov, 'mean_base': means, 'mean_scal': w}
                    else:
                        hycprob[str(key_comb)] = {'variance': np.nan, 'mean_base': np.nan, 'mean_scal': []}
            return {"hybcprob": hycprob}


class LogitNode(DiscreteNode):
    def __init__(self, name):
        super(LogitNode, self).__init__(name)
        self.type = 'Logit'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.cont_parents
        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
        model.fit(data[parents].values, data[self.name].values)
        return {"mean_base": list(model.intercept_),
                "mean_scal": list(model.coef_.reshape(1, -1)[0])}


class ConditionalLogitNode(DiscreteNode):
    def __init__(self, name):
        super(ConditionalLogitNode, self).__init__(name)
        self.type = 'ConditionalLogit'

    def fit_parameters(self, data):
        parents = self.disc_parents + self.cont_parents
        hycprob = dict()
        values = []
        combinations = []
        for d_p in self.disc_parents:
            values.append(np.unique(data[d_p].values))
        for xs in itertools.product(*values):
            combinations.append(list(xs))
        for comb in combinations:
            mask = np.full(len(data), True)
            for col, val in zip(self.disc_parents, comb):
                mask = (mask) & (data[col] == val)
            new_data = data[mask]
            mean_base = [np.nan]
            classes = []
            if new_data.shape[0] != 0:
                model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                values = set(new_data[self.name])
                if len(values) > 1:
                    model.fit(new_data[self.cont_parents].values, new_data[self.name].values)
                    mean_base = model.intercept_
                    classes = model.classes_
                    coef = model.coef_
                else:
                    mean_base = np.array([0.0])
                    coef = np.array([float_info.max])
                    classes = list(values)

                key_comb = [str(x) for x in comb]
                hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base),
                                          'mean_scal': list(coef.reshape(1, -1)[0])}
            else:
                scal = list(np.full(len(self.cont_parents), np.nan))
                key_comb = [str(x) for x in comb]
                hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base), 'mean_scal': scal}
        return {"hybcprob": hycprob}
