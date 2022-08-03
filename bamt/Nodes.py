from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from concurrent.futures import ThreadPoolExecutor
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
from bamt.utils.MathUtils import *
from gmr import GMM
# from sklearn.mixture import GaussianMixture
from pandas import DataFrame

from typing import Dict, List, Any, Union, TypedDict, Type, Optional, Sequence

import itertools
import random
import joblib
import pickle
import os

from bamt.log import logger_nodes
from bamt.config import config

STORAGE = config.get('NODES', 'models_storage', fallback='models_storage is not defined')


class DiscreteParams(TypedDict):
    cprob: Union[List[Union[list, Any]], Dict[str, list]]
    vals: List[str]


class BaseNode(object):
    """
    Base class for nodes.
    """

    def __init__(self, name: str):
        """
        :param name: name for node (taken from column name)
        type: node type
        disc_parents: list with discrete parents
        cont_parents: list with continuous parents
        children: node's children
        """
        self.name = name
        self.type = 'abstract'

        self.disc_parents = []
        self.cont_parents = []
        self.children = []

    def __repr__(self):
        return f"{self.name}"

    @staticmethod
    def choose_serialization(model) -> Union[str, Exception]:
        try:
            ex_b = pickle.dumps(model, protocol=4)
            model_ser = ex_b.decode('latin1').replace('\'', '\"')
            a = model_ser.replace('\"', '\'').encode('latin1')
            classifier_body = pickle.loads(a)
            return 'pickle'
        except Exception as ex:
            return ex


class DiscreteNode(BaseNode):
    """
    Main class of Discrete Node
    """

    def __init__(self, name):
        super(DiscreteNode, self).__init__(name)
        self.type = 'Discrete'

    def fit_parameters(self, data: DataFrame, num_workers: int = 1):
        """
        Train params for Discrete Node
        data: DataFrame to train on
        num_workers: number of Parallel Workers
        Method returns probas dict with following format {[<combinations>: value]}
        and vals, list of appeared values in combinations
        """

        def worker(node: Type[BaseNode]) -> DiscreteParams:
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

        pool = ThreadPoolExecutor(num_workers)
        future = pool.submit(worker, self)
        return future.result()

    @staticmethod
    def choose(node_info: Dict[str, Union[float, str]], pvals: List[str]) -> str:
        """
        Return value from discrete node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        rindex = 0
        random.seed()
        vals = node_info['vals']
        if not pvals:
            dist = node_info['cprob']
        else:
            # noinspection PyTypeChecker
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

    @staticmethod
    def predict(node_info: Dict[str, Union[float, str]], pvals: List[str]) -> str:
        """function for prediction based on evidence values in discrete node

        Args:
            node_info (Dict[str, Union[float, str]]): parameters of node
            pvals (List[str]): values in parents nodes

        Returns:
            str: prediction
        """        
        # vals = node_info['vals']
        # if not pvals:
        #     dist = node_info['cprob']
        # else:
        #     # noinspection PyTypeChecker
        #     dist = node_info['cprob'][str(pvals)]
        # index_max = 0    
        # g = itertools.groupby(dist)
        # if next(g, True) and not next(g, False):
        #     index_max = random.randint(0, len(dist))
        # else:
        #     index_max = np.argmax(dist)
        # return vals[index_max]

        vals = node_info['vals']
        disct = []
        if not pvals:
            dist = node_info['cprob']
        else:
            # noinspection PyTypeChecker
            dist = node_info['cprob'][str(pvals)]
        max_value = max(dist)
        indices = [index for index, value in enumerate(dist) if value == max_value]
        max_ind = 0
        if len(indices) == 1:
            max_ind = indices[0]
        else:
            max_ind = random.choice(indices)
        return vals[max_ind]


class GaussianParams(TypedDict):
    mean: np.ndarray
    coef: Sequence[Any]
    variance: Union[np.ndarray, float]


class GaussianNode(BaseNode):
    """
    Main class for Gaussian Node
    """

    def __init__(self, name: str, model=linear_model.LinearRegression()):
        super(GaussianNode, self).__init__(name)
        self.model = model
        self.type = 'Gaussian'

    def fit_parameters(self, data: DataFrame) -> GaussianParams:
        """
        Function for training parameters for gaussian node
        """
        parents = self.disc_parents + self.cont_parents
        if parents:
            # model = self.model
            predict = []
            if len(parents) == 1:
                self.model.fit(np.transpose([data[parents[0]].values]), data[self.name].values)
                predict = self.model.predict(np.transpose([data[parents[0]].values]))
            else:
                self.model.fit(data[parents].values, data[self.name].values)
                predict = self.model.predict(data[parents].values)
            variance = mse(data[self.name].values, predict)
            return {"mean": self.model.intercept_,
                    "coef": list(self.model.coef_),
                    "variance": variance}
        else:
            mean_base = np.mean(data[self.name].values)
            self.model.intercept_ = mean_base
            self.model.coef_ = np.array([])
            variance = np.var(data[self.name].values)
            return {"mean": mean_base,
                    "coef": [],
                    "variance": variance}

    @staticmethod
    def choose(node_info: Dict[str, Union[float, List[float]]],
               pvals: List[float]) -> float:
        """
        Return value from Gaussian node
        params:
        node_info: information about node
        pvals: parent values
        """
        mean = node_info["mean"]
        if pvals:
            for i, m in enumerate(pvals):
                mean += m * node_info['coef'][i]
        variance = node_info['variance']
        # distribution = [mean, variance]
        return random.gauss(mean, math.sqrt(variance))

    @staticmethod
    def predict(node_info: Dict[str, Union[float, List[float]]],
                pvals: List[float]) -> float:
        """function for prediction in gaussian node

        Args:
            node_info (Dict[str, Union[float, List[float]]]): node parameters
            pvals (List[float]): parent values

        Returns:
            float: prediction
        """
        mean = node_info["mean"]
        if pvals:
            for i, m in enumerate(pvals):
                mean += m * node_info['coef'][i]
        return mean


class CondGaussParams(TypedDict):
    variance: Optional[float]
    mean: Optional[List[float]]
    coef: List[float]


class ConditionalGaussianNode(BaseNode):
    """
    Main class for Conditional Gaussian Node
    """

    def __init__(self, name):
        super(ConditionalGaussianNode, self).__init__(name)
        self.type = 'ConditionalGaussian'

    def fit_parameters(self, data: DataFrame) -> Dict[str, Dict[str, CondGaussParams]]:
        """
        Train params for Conditional Gaussian Node.
        Return:
        {"hybcprob": {<combination of outputs from discrete parents> : CondGaussParams}}
        """
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
            key_comb = [str(x) for x in comb]
            if new_data.shape[0] != 0:
                if self.cont_parents:
                    model = linear_model.LinearRegression()
                    if len(self.cont_parents) == 1:
                        model.fit(np.transpose([new_data[self.cont_parents[0]].values]), new_data[self.name].values)
                        predict = model.predict(np.transpose([new_data[self.cont_parents[0]].values]))
                    else:
                        model.fit(new_data[self.cont_parents].values, new_data[self.name].values)
                        predict = model.predict(new_data[self.cont_parents].values)
                    mean_base = model.intercept_
                    variance = mse(new_data[self.name].values, predict)
                    hycprob[str(key_comb)] = {'variance': variance, 'mean': mean_base,
                                              'coef': list(model.coef_)}
                else:
                    mean_base = np.mean(new_data[self.name].values)
                    variance = np.var(new_data[self.name].values)
                    hycprob[str(key_comb)] = {'variance': variance, 'mean': mean_base, 'coef': []}
            else:
                if self.cont_parents:
                    scal = list(np.full(len(self.cont_parents), np.nan))
                    hycprob[str(key_comb)] = {'variance': variance, 'mean': mean_base, 'coef': scal}
                else:
                    # mean_base = np.nan
                    # variance = np.nan
                    hycprob[str(key_comb)] = {'variance': variance, 'mean': mean_base, 'coef': []}
        return {"hybcprob": hycprob}

    @staticmethod
    def choose(node_info: Dict[str, Dict[str, CondGaussParams]], pvals: List[Union[str, float]]) -> float:
        """
        Return value from ConditionalGaussian node
        node_info: nodes info from distributions
        pvals: parent values
        """
        dispvals = []
        lgpvals = []
        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        if lgpvals:
            for x in range(len(lgpvals)):
                mean += lgpvals[x] * lgdistribution["coef"][x]
        variance = lgdistribution["variance"]
        return random.gauss(mean, math.sqrt(variance))

    @staticmethod
    def predict(node_info: Dict[str, Dict[str, CondGaussParams]], pvals: List[Union[str, float]]) -> float:
        """function for prediction in conditional gaussian node

        Args:
            node_info (Dict[str, Union[float, List[float]]]): node parameters
            pvals (List[float]): parent values

        Returns:
            float: prediction
        """
        dispvals = []
        lgpvals = []
        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        if lgpvals:
            for x in range(len(lgpvals)):
                mean += lgpvals[x] * lgdistribution["coef"][x]
        return mean


class MixtureGaussianParams(TypedDict):
    mean: List[float]
    coef: List[float]
    covars: List[float]


class MixtureGaussianNode(BaseNode):
    """
    Main class for Mixture Gaussian Node
    """

    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = 'MixtureGaussian'

    def fit_parameters(self, data: DataFrame) -> MixtureGaussianParams:
        """
        Train params for Mixture Gaussian Node
        """
        parents = self.disc_parents + self.cont_parents
        if not parents:
            n_comp = int((component(data, [self.name], 'aic') + component(data, [self.name],
                                                                          'bic')) / 2)  # component(data, [node], 'LRTS')#
            # n_comp = 3
            gmm = GMM(n_components=n_comp).from_samples(np.transpose([data[self.name].values]), n_iter=500,
                                                        init_params='kmeans++')
            means = gmm.means.tolist()
            cov = gmm.covariances.tolist()
            # weigts = np.transpose(gmm.to_responsibilities(np.transpose([data[node].values])))
            w = gmm.priors.tolist()  # []
            # for row in weigts:
            #     w.append(np.mean(row))
            return {"mean": means, "coef": w, "covars": cov}
        if parents:
            if not self.disc_parents and self.cont_parents:
                nodes = [self.name] + self.cont_parents
                new_data = data[nodes]
                new_data.reset_index(inplace=True, drop=True)
                n_comp = int((component(new_data, nodes, 'aic') + component(new_data, nodes,
                                                                            'bic')) / 2)  # component(new_data, nodes, 'LRTS')#
                # n_comp = 3
                gmm = GMM(n_components=n_comp).from_samples(new_data[nodes].values, n_iter=500, init_params='kmeans++')
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                return {"mean": means,
                        "coef": w,
                        "covars": cov}

    @staticmethod
    def choose(node_info: MixtureGaussianParams, pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Func to get value from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info['coef'])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                    cond_gmm = gmm.condition(indexes, [pvals])
                    sample = cond_gmm.sample(1)[0][0]
                else:
                    sample = np.nan
            else:
                gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = gmm.sample(1)[0][0]
        else:
            sample = np.nan
        return sample

    @staticmethod
    def predict(node_info: MixtureGaussianParams, pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Func to get prediction from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info['coef'])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                    sample = gmm.predict(indexes, [pvals])[0][0]
                else:
                    sample = np.nan
            else:
                # gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
        else:
            sample = np.nan
        return sample


class CondMixtureGaussParams(TypedDict):
    mean: Optional[List[float]]
    coef: List[float]
    covars: Optional[List[float]]


class ConditionalMixtureGaussianNode(BaseNode):
    """
    Main class for Conditional Mixture Gaussian Node
    """

    def __init__(self, name):
        super(ConditionalMixtureGaussianNode, self).__init__(name)
        self.type = 'ConditionalMixtureGaussian'

    def fit_parameters(self, data: DataFrame) -> Dict[str, Dict[str, CondMixtureGaussParams]]:
        """
        Train params for Conditional Mixture Gaussian Node.
        Return:
        {"hybcprob": {<combination of outputs from discrete parents> : CondMixtureGaussParams}}
        """
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
                if self.cont_parents:
                    n_comp = int((component(new_data, nodes, 'aic') + component(new_data, nodes,
                                                                                'bic')) / 2)  # component(new_data, nodes, 'LRTS')#int((component(new_data, nodes, 'aic') + component(new_data, nodes, 'bic')) / 2)
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp).from_samples(new_data[nodes].values, n_iter=500,
                                                                init_params='kmeans++')
                else:
                    n_comp = int((component(new_data, [self.name], 'aic') + component(new_data, [self.name],
                                                                                      'bic')) / 2)  # component(new_data, [node], 'LRTS')#int((component(new_data, [node], 'aic') + component(new_data, [node], 'bic')) / 2)
                    # n_comp = 3
                    gmm = GMM(n_components=n_comp).from_samples(np.transpose([new_data[self.name].values]), n_iter=500,
                                                                init_params='kmeans++')
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                hycprob[str(key_comb)] = {'covars': cov, 'mean': means, 'coef': w}
            elif new_data.shape[0] != 0:
                n_comp = 1
                gmm = GMM(n_components=n_comp)
                if self.cont_parents:
                    gmm.from_samples(new_data[nodes].values)
                else:
                    gmm.from_samples(np.transpose([new_data[self.name].values]))
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(np.transpose([new_data[node].values])))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                hycprob[str(key_comb)] = {'covars': cov, 'mean': means, 'coef': w}
            else:
                if self.cont_parents:
                    hycprob[str(key_comb)] = {'covars': np.nan, 'mean': np.nan, 'coef': []}
                else:
                    hycprob[str(key_comb)] = {'covars': np.nan, 'mean': np.nan, 'coef': []}
        return {"hybcprob": hycprob}

    @staticmethod
    def choose(node_info: Dict[str, Dict[str, CondMixtureGaussParams]],
               pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Function to get value from ConditionalMixtureGaussian node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        dispvals = []
        lgpvals = []
        for pval in pvals:
            if ((isinstance(pval, str)) | ((isinstance(pval, int)))):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]
        if len(w) != 0:
            if len(lgpvals) != 0:
                indexes = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    n_comp = len(w)
                    gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                    cond_gmm = gmm.condition(indexes, [lgpvals])
                    sample = cond_gmm.sample(1)[0][0]
                else:
                    sample = np.nan
            else:
                n_comp = len(w)
                gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = gmm.sample(1)[0][0]
        else:
            sample = np.nan
        return sample

    @staticmethod
    def predict(node_info: Dict[str, Dict[str, CondMixtureGaussParams]],
                pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Function to get prediction from ConditionalMixtureGaussian node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        dispvals = []
        lgpvals = []
        for pval in pvals:
            if ((isinstance(pval, str)) | ((isinstance(pval, int)))):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]
        if len(w) != 0:
            if len(lgpvals) != 0:
                indexes = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    n_comp = len(w)
                    gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                    sample = gmm.predict(indexes, [lgpvals])[0][0]
                else:
                    sample = np.nan
            else:
                # n_comp = len(w)
                # gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
        else:
            sample = np.nan
        return sample


class LogitParams(TypedDict):
    classes: List[int]
    classifier: str
    classifier_obj: Optional[Union[str, bool, bytes]]
    serialization: str


class LogitNode(BaseNode):
    """
    Main class for logit node
    """

    def __init__(self, name, classifier: Optional[object] = None):
        super(LogitNode, self).__init__(name)
        if classifier is None:
            classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
        self.classifier = classifier
        self.type = 'Logit' + f" ({type(self.classifier).__name__})"

    def fit_parameters(self, data: DataFrame) -> LogitParams:
        parents = self.disc_parents + self.cont_parents
        self.classifier.fit(data[parents].values, data[self.name].values)
        serialization = self.choose_serialization(self.classifier)

        if serialization == 'pickle':
            ex_b = pickle.dumps(self.classifier, protocol=4)
            # model_ser = ex_b.decode('latin1').replace('\'', '\"')
            model_ser = ex_b.decode('latin1')
            return {'classes': list(self.classifier.classes_),
                    'classifier_obj': model_ser,
                    'classifier': type(self.classifier).__name__,
                    'serialization': 'pickle'}
        else:
            logger_nodes.warning(f"{self.name}::Pickle failed. BAMT will use Joblib. | " + str(serialization.args[0]))
            index = str(int(os.listdir(STORAGE)[-1]))
            if not os.path.isdir(os.path.join(STORAGE, index, f"{self.name.replace(' ', '_')}")):
                os.makedirs(os.path.join(STORAGE, index, f"{self.name.replace(' ', '_')}"))
            path = os.path.abspath(os.path.join(STORAGE,
                                                index,
                                                f"{self.name.replace(' ', '_')}",
                                                f"{self.name.replace(' ', '_')}.joblib.compressed"))

            joblib.dump(self.classifier, path, compress=True, protocol=4)
            return {'classes': list(self.classifier.classes_),
                    'classifier_obj': path,
                    'classifier': type(self.classifier).__name__,
                    'serialization': 'joblib'}

    def choose(self, node_info: LogitParams, pvals: List[Union[str, float]]) -> str:
        """
        Return value from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        pvals = [str(p) for p in pvals]

        rindex = 0

        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == 'joblib':
                model = joblib.load(node_info["classifier_obj"])
            else:
                # str_model = node_info["classifier_obj"].decode('latin1').replace('\'', '\"')
                a = node_info["classifier_obj"].encode('latin1')
                model = pickle.loads(a)

            distribution = model.predict_proba(np.array(pvals).reshape(1, -1))[0]

            # choose
            rand = random.random()
            lbound = 0
            ubound = 0
            for interval in range(len(node_info["classes"])):
                ubound += distribution[interval]
                if (lbound <= rand and rand < ubound):
                    rindex = interval
                    break
                else:
                    lbound = ubound

            return str(node_info["classes"][rindex])

        else:
            return str(node_info["classes"][0])

    def predict(self, node_info: LogitParams, pvals: List[Union[str, float]]) -> str:
        """
        Return prediction from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        pvals = [str(p) for p in pvals]

        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == 'joblib':
                model = joblib.load(node_info["classifier_obj"])
            else:
                # str_model = node_info["classifier_obj"].decode('latin1').replace('\'', '\"')
                a = node_info["classifier_obj"].encode('latin1')
                model = pickle.loads(a)

            pred = model.predict(np.array(pvals).reshape(1, -1))[0]

            return str(pred)

        else:
            return str(node_info["classes"][0])


class ConditionalLogitNode(BaseNode):
    """
    Main class for Conditional Logit Node
    """

    def __init__(self, name: str, classifier: Optional[object] = None):
        super(ConditionalLogitNode, self).__init__(name)
        if classifier is None:
            classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
        self.classifier = classifier
        self.type = 'ConditionalLogit' + f" ({type(self.classifier).__name__})"

    def fit_parameters(self, data: DataFrame) -> Dict[str, Dict[str, LogitParams]]:
        """
        Train params on data
        Return:
        {"hybcprob": {<combination of outputs from discrete parents> : LogitParams}}
        """
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
            # mean_base = [np.nan]
            classes = []
            key_comb = [str(x) for x in comb]
            if new_data.shape[0] != 0:
                model = self.classifier
                values = set(new_data[self.name])
                if len(values) > 1:
                    model.fit(new_data[self.cont_parents].values, new_data[self.name].values)
                    classes = list(model.classes_)
                    serialization = self.choose_serialization(model)

                    if serialization == 'pickle':
                        ex_b = pickle.dumps(self.classifier, protocol=4)
                        model_ser = ex_b.decode('latin1')

                        # model_ser = pickle.dumps(self.classifier, protocol=4)
                        hycprob[str(key_comb)] = {'classes': classes,
                                                  'classifier_obj': model_ser,
                                                  'classifier': type(self.classifier).__name__,
                                                  'serialization': 'pickle'}
                    else:
                        logger_nodes.warning(
                            f"{self.name} {comb}::Pickle failed. BAMT will use Joblib. | " + str(serialization.args[0]))
                        index = str(int(os.listdir(STORAGE)[-1]))
                        if not os.path.isdir(os.path.join(STORAGE, index, f"{self.name.replace(' ', '_')}")):
                            os.makedirs(os.path.join(STORAGE, index, f"{self.name.replace(' ', '_')}"))
                        path = os.path.abspath(os.path.join(STORAGE,
                                                            index,
                                                            f"{self.name.replace(' ', '_')}",
                                                            f"{comb}.joblib.compressed"))

                        joblib.dump(model, path, compress=True, protocol=4)
                        hycprob[str(key_comb)] = {'classes': classes,
                                                  'classifier_obj': path,
                                                  'classifier': type(self.classifier).__name__,
                                                  'serialization': 'joblib'}
                else:
                    classes = list(values)
                    hycprob[str(key_comb)] = {'classes': classes, 'classifier': type(self.classifier).__name__,
                                              'classifier_obj': None, 'serialization': None}

            else:
                hycprob[str(key_comb)] = {'classes': list(classes), 'classifier': type(self.classifier).__name__,
                                          'classifier_obj': None, 'serialization': None}
        return {"hybcprob": hycprob}

    def choose(self, node_info: Dict[str, Dict[str, LogitParams]], pvals: List[Union[str, float]]) -> str:
        """
        Return value from ConditionalLogit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        dispvals = []
        lgpvals = []
        for pval in pvals:
            if (isinstance(pval, str)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)

        lgdistribution = node_info["hybcprob"][str(dispvals)]

        # JOBLIB
        if len(lgdistribution["classes"]) > 1:
            if lgdistribution["serialization"] == 'joblib':
                model = joblib.load(lgdistribution["classifier_obj"])
            else:
                # str_model = lgdistribution["classifier_obj"].decode('latin1').replace('\'', '\"')
                bytes_model = lgdistribution["classifier_obj"].encode('latin1')
                model = pickle.loads(bytes_model)

            distribution = model.predict_proba(np.array(lgpvals).reshape(1, -1))[0]

            rand = random.random()
            rindex = 0
            lbound = 0
            ubound = 0
            for interval in range(len(lgdistribution["classes"])):
                ubound += distribution[interval]
                if (lbound <= rand and rand < ubound):
                    rindex = interval
                    break
                else:
                    lbound = ubound
            return str(lgdistribution["classes"][rindex])

        else:
            return str(lgdistribution["classes"][0])

    def predict(self, node_info: Dict[str, Dict[str, LogitParams]], pvals: List[Union[str, float]]) -> str:
        """
        Return value from ConditionalLogit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        dispvals = []
        lgpvals = []
        for pval in pvals:
            if (isinstance(pval, str)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)

        lgdistribution = node_info["hybcprob"][str(dispvals)]

        # JOBLIB
        if len(lgdistribution["classes"]) > 1:
            if lgdistribution["serialization"] == 'joblib':
                model = joblib.load(lgdistribution["classifier_obj"])
            else:
                # str_model = lgdistribution["classifier_obj"].decode('latin1').replace('\'', '\"')
                bytes_model = lgdistribution["classifier_obj"].encode('latin1')
                model = pickle.loads(bytes_model)

            pred = model.predict(np.array(lgpvals).reshape(1, -1))[0]

            return str(pred)

        else:
            return str(lgdistribution["classes"][0])
