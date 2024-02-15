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

disc_network_params = []
for opt in optimizer:
    for score in scoring:
        if opt == 'HC':
            disc_network_params.append(
                (discrete_data_path, False, False, opt, score, "Discrete", 'Tectonic regime'))
disc_network_params.append((discrete_data_path, False, False,
                           'Evo', ("K2", K2Score), "Discrete", 'Tectonic regime'))

cont_network_params = []
for opt in optimizer:
    for score in scoring:
        for mix in use_mixture:
            if opt == 'HC':
                cont_network_params.append(
                    (continuous_data_path, mix, False, opt, score, "Continuous", 'target'))
cont_network_params.append(
    (continuous_data_path, mix, False, 'Evo', ("K2", K2Score), "Continuous", 'target'))

hybrid_network_params = []
for opt in optimizer:
    for score in scoring:
        for mix in use_mixture:
            for logit in has_logit:
                if opt == 'HC':
                    hybrid_network_params.append(
                        (hybrid_data_path, mix, logit, opt, score, "Hybrid", 'target'))
hybrid_network_params.append(
    (hybrid_data_path, mix, logit, 'Evo', ("K2", K2Score), "Hybrid", 'target'))

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
    # Проверка равенства предсказаний (обученной и загруженной сети) до и после сохранения
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

    # Проверка равенсвта предсказаний (обученной сети) до и после сохренения
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

    # Проверка предсказаний сети без рёбер
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

    # Проверка сети обученной на 1-ом сэмпле
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
