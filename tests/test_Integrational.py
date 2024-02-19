import itertools
import pytest
import bamt.networks as networks
import bamt.preprocessors as pp
from pgmpy.estimators import K2Score
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

continuous_data_path = 'data/benchmark/auto_price.csv'
discrete_data_path = 'tests/hack_discrete/hack_data.csv'
hybrid_data_path = 'data/benchmark/new_thyroid.csv'

scoring = [("K2", K2Score), 'BIC', 'MI']
optimizer = ['HC', 'Evo']
use_mixture = [True, False]
has_logit = [True, False]

# Generate parameter lists using list comprehension and itertools.product
def generate_params(data_path, data_type, additional_params=None):
    params = []
    for opt, score in itertools.product(optimizer, scoring):
        if opt == 'HC':
            base_params = [data_path, False, False, opt, score, data_type]
            if additional_params:
                params.extend([base_params + list(param) for param in itertools.product(*additional_params)])
            else:
                params.append(base_params)
    # Add the 'Evo' optimizer case
    params.append([data_path] + [False, False, 'Evo', ("K2", K2Score), data_type] + (list(additional_params[0]) if additional_params else []))
    return params

disc_network_params = generate_params(discrete_data_path, "Discrete", [('Tectonic regime',)])
cont_network_params = generate_params(continuous_data_path, "Continuous", [(use_mixture, 'target')])
hybrid_network_params = generate_params(hybrid_data_path, "Hybrid", [(use_mixture, has_logit, 'target')])

composite_network_params = [
    (hybrid_data_path, False, False, 'Evo', ("K2", K2Score), "Composite", 'target')]

params = disc_network_params + cont_network_params + hybrid_network_params


def initialize_bn(bn_type, use_mixture, has_logit):
    if bn_type == "Discrete":
        bn = networks.DiscreteBN()
    elif bn_type == "Continuous":
        bn = networks.ContinuousBN(use_mixture=use_mixture)
    elif bn_type == "Hybrid":
        bn = networks.HybridBN(has_logit=has_logit, use_mixture=use_mixture)
    elif bn_type == "Composite":
        bn = networks.CompositeBN()
    return bn


def prepare_data(directory):
    data = pd.read_csv(directory, index_col=0)
    train, test = train_test_split(data, test_size=0.33, random_state=42)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(
        n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(train)
    info = p.info
    return info, discretized_data, train, test


class TestNetwork:
    # Checking the equality of predictions (trained and loaded network) before and after saving
    @pytest.mark.parametrize("directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params)
    def test_1(self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target):
        test_id = 'test_1'

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        if bn_type != "Composite":
            bn.add_edges(discretized_data, optimizer=optimizer,
                         scoring_function=scoring, progress_bar=False,)
        else:
            bn.add_edges(train)

        bn.fit_parameters(train)
        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False)
        bn.save('bn')

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        bn.load('bn.json')
        predict_loaded = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False)

        try:
            assert_frame_equal(pd.DataFrame(predict),
                               pd.DataFrame(predict_loaded))
            print(f'{test_id} runned successfully')
        except Exception as e:
            print(e)
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}")
            raise

    # Checking the prediction algorithm (trained network) before and after saving
    @pytest.mark.parametrize("directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params)
    def test_2(self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target):
        test_id = 'test_2'

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        if bn_type != "Composite":
            bn.add_edges(discretized_data, optimizer=optimizer,
                         scoring_function=scoring, progress_bar=False,)
        else:
            bn.add_edges(train)

        bn.fit_parameters(train)
        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False)
        bn.save('bn')

        predict2 = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False)

        try:
            assert_frame_equal(pd.DataFrame(predict), pd.DataFrame(predict2))
            print(f'{test_id} runned successfully')
        except Exception as e:
            print(e)
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}")
            raise

    # Checking network predictions without edges
    @pytest.mark.parametrize("directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params)
    def test_3(self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target):
        test_id = 'test_3'

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        bn.fit_parameters(train)

        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False)

        try:
            if info['types'][target] == 'cont':
                if use_mixture:
                    mean = bn.distributions[target]['mean']
                    w = bn.distributions[target]['coef']
                    sample = 0
                    for ind, wi in enumerate(w):
                        sample += wi * mean[ind][0]
                else:
                    sample = train[target].mean()

                for x in predict[target]:
                    assert x == sample

            elif info['types'][target] == 'disc_num':
                most_frequent = train[target].value_counts().index[0]
                for x in predict[target]:
                    assert x == most_frequent
            print(f'{test_id} runned successfully')
        except Exception as e:
            print(e)
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}")
            raise

    # Checking the network trained on the 1 sample
    @pytest.mark.parametrize("directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params)
    def test_4(self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target):
        test_id = 'test_3'

        if use_mixture == False:

            bn = initialize_bn(bn_type, use_mixture, has_logit)
            info, discretized_data, train, test = prepare_data(directory)

            bn.add_nodes(info)

            train_data_1 = pd.DataFrame(train.iloc[0].to_dict(), index=[0])
            disc_data_1 = pd.DataFrame(
                discretized_data.iloc[0].to_dict(), index=[0])

            if bn_type != "Composite":
                bn.add_edges(disc_data_1, optimizer=optimizer,
                             scoring_function=scoring, progress_bar=False,)
            else:
                bn.add_edges(train_data_1)

            bn.fit_parameters(train_data_1)

            predict = bn.predict(
                test[[x for x in test.columns if x != target]], progress_bar=False)

            try:
                for x in predict[target]:
                    assert x == train_data_1[target][0]
                print(f'{test_id} runned successfully')
            except Exception as e:
                print(e)
                print(
                    f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}")
                raise
        else:
            pass
