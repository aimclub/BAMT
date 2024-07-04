import copy
import math
from typing import List, Any, Union, Dict

from golem.utilities.requirements_notificator import warn_requirement

try:
    import torch
except ModuleNotFoundError:
    warn_requirement('torch', 'other_requirements/adaptive.txt')

import numpy as np
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.utils import Arm, Constants, Num

from golem.core.log import default_log

import warnings

warnings.filterwarnings("ignore")


class NeuralMAB(MAB):
    """
    Neural Multi-Armed Bandit.
    The main concept is explained in the article: https://arxiv.org/abs/2012.01780.
    Deep representation is formed with NN and Contextual Multi-Armed Bandit is integrated to choose arm.

    NB! Neural MABs can be used with 1.8.0 version of torch since some methods are deprecated in later versions,
    however, python of version 3.10 is not supported in this version of torch.
    """

    def __init__(self, arms: List[Arm],
                 learning_policy: Any = LearningPolicy.UCB1(alpha=1.25),
                 neighborhood_policy: Any = NeighborhoodPolicy.Clusters(),
                 seed: int = Constants.default_seed,
                 n_jobs: int = 1):

        super().__init__(arms, learning_policy, neighborhood_policy, seed, n_jobs)
        self.nn_with_se = NNWithShallowExploration(context_size=1, arms_count=len(arms))
        self.arms = arms
        self.seed = seed
        self.n_jobs = n_jobs
        self.log = default_log('NeuralMAB')
        # to track when GNN needs to be updated
        self.iter = 0
        self._indices = list(range(len(arms)))
        self._mab = MAB(arms=self._indices,
                        learning_policy=learning_policy,
                        neighborhood_policy=neighborhood_policy,
                        n_jobs=n_jobs)
        self.is_fitted = False

    def _initial_fit_mab(self, context: Any):
        """ Initial fit for Contextual Multi-Armed Bandit.
        At this step, all hands are assigned the same weights with the very first context
        that is fed to the bandit. """
        # initial fit for mab
        n = len(self.arms)
        uniform_rewards = [1. / n] * n
        deep_context = self._get_deep_context(context=context)
        self._mab.fit(decisions=self._indices, rewards=uniform_rewards, contexts=n * [deep_context])
        self.is_fitted = True

    def partial_fit(self, decisions: List[Any], rewards: List[float], contexts: List[Any] = None):

        # get deep contexts, calculate regret and update weights for NN (once in _H_q iters)
        deep_contexts = self.nn_with_se.partial_fit(iter=self.iter, decisions=decisions,
                                                    rewards=rewards, contexts=contexts)
        self.iter += 1

        # update contextual mab with deep contexts
        self._mab.partial_fit(decisions=decisions, contexts=deep_contexts, rewards=rewards)

    def predict(self, contexts: Any = None) -> Union[Arm, List[Arm]]:
        """ Predicts which arm to pull to get maximum reward. """
        if not self.is_fitted:
            self._initial_fit_mab(context=contexts)
        deep_context = self._get_deep_context(context=contexts)
        a_choose = self._mab.predict(contexts=[deep_context])
        return a_choose

    def predict_expectations(self, contexts: Any = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        """ Returns expected reward for each arm. """
        if not self.is_fitted:
            self._initial_fit_mab(context=contexts)
        deep_context = self._get_deep_context(context=contexts)
        return self._mab.predict_expectations(contexts=[deep_context])

    def _get_deep_context(self, context: Any) -> List[Any]:
        """ Returns deep representation of context. """
        temp = self.nn_with_se.transfer(context, 0, len(self.arms))
        feat = self.nn_with_se.feature_extractor(temp, self.nn_with_se.W).numpy().squeeze()
        return list(feat)


class NNWithShallowExploration:
    """ Neural Network with shallow exploration which means that weights are updated every _H_q iterations. """

    def __init__(self, context_size: int, arms_count: int):
        """
        Initial fit for NN.
        beta -- parameter for UCB exploration
        H_q -- how many time steps to update NN
        interT -- internal steps for GD
        """
        self._beta = 0.02
        self._lambd = 1
        self._lr = 0.001
        self._H_q = 5
        self._interT = 1000
        self._hidden_dim = [1000, 1000]
        hid_dim_lst = self._hidden_dim
        dim_second_last = self._hidden_dim[-1] * 2

        dim_for_init = [context_size + arms_count] + hid_dim_lst + [1]
        self.arms_count = arms_count
        self.W0, total_dim = self._initialization(dim_for_init)
        self.LAMBDA = self._lambd * torch.eye(dim_second_last, dtype=torch.double)
        self.bb = torch.zeros(self.LAMBDA.size()[0], dtype=torch.double).view(-1, 1)

        theta = np.random.randn(dim_second_last, 1) / np.sqrt(dim_second_last)
        self.theta = torch.from_numpy(theta)

        self.THETA_action = torch.tensor([])
        self.CONTEXT_action = torch.tensor([])
        self.REWARD_action = torch.tensor([])
        self.result_neuralucb = []
        self.W = copy.deepcopy(self.W0)
        self.summ = 0
        self.log = default_log('NNWithShallowExploration')

    def partial_fit(self, iter: int,
                    decisions: List[Any], rewards: List[float], contexts: List[Any] = None):
        deep_contexts = []

        # update NN and calculate reward
        for decision, context, reward in zip(decisions, contexts, rewards):

            # calculate reward
            temp = self.transfer(context, decision, self.arms_count)
            feat = self.feature_extractor(temp, self.W)
            deep_contexts.append(list(feat.numpy().squeeze()))
            expected_reward = torch.mm(self.theta.view(1, -1), feat) + self._beta * self.UCB(self.LAMBDA, feat)

            self.summ += (expected_reward - reward)
            self.result_neuralucb.append(self.summ)

            # gather dataset for next NN training (context_action and reward_action)
            if np.mod(iter, self._H_q) == 0:
                context_action = temp
                reward_action = torch.tensor([reward], dtype=torch.double)
            else:
                context_action = torch.cat((self.CONTEXT_action, temp), 1)
                reward_action = torch.cat((self.REWARD_action, torch.tensor([reward], dtype=torch.double)), 0)

            # update LAMBDA and bb
            self.LAMBDA += torch.mm(self.feature_extractor(temp, self.W),
                                    self.feature_extractor(temp, self.W).t())
            self.bb += reward * self.feature_extractor(temp, self.W)
            theta, _ = torch.solve(self.bb, self.LAMBDA)

            if np.mod(iter, self._H_q) == 0:
                theta_action = theta.view(-1, 1)
            else:
                theta_action = torch.cat((self.THETA_action, theta.view(-1, 1)), 1)

        # update weight of NN
        if np.mod(iter, self._H_q) == self._H_q - 1:
            self.log.info(f'Current regret: {self.summ}')
            self.W = self.train_with_shallow_exploration(context_action, reward_action, self.W0,
                                                         self._interT, self._lr, theta_action, self._H_q)
        return deep_contexts

    @staticmethod
    def UCB(A, phi):
        """ Ucb term. """
        try:
            tmp, _ = torch.solve(phi, A)
        except Exception:
            tmp = torch.Tensor(np.linalg.solve(A, phi))

        return torch.sqrt(torch.mm(torch.transpose(phi, 0, 1).double(), tmp.double()))

    @staticmethod
    def transfer(c, a, arm_size):
        """
        Transfer an array context + action to new context with dimension 2*(__context__ + __armsize__).
        """
        action = np.zeros(arm_size)
        action[a] = 1
        c_final = np.append(c, action)
        c_final = torch.from_numpy(c_final)
        c_final = c_final.view((len(c_final), 1))
        c_final = c_final.repeat(2, 1)
        return c_final

    def train_with_shallow_exploration(self, X, Y, W_start, T, et, THETA, H):
        """ Gd-based model training with shallow exploration
        Dataset X, label Y. """
        W = copy.deepcopy(W_start)
        X = X[:, -H:]
        Y = Y[-H:]
        THETA = THETA[:, -H:]

        prev_loss = 1000000
        prev_loss_1k = 1000000
        for i in range(0, T):
            grad = self._gradient_loss(X, Y, W, THETA)
            for j in range(0, len(W) - 1):
                W[j] = W[j] - et * grad[j]

            curr_loss = self._loss(X, Y, W, THETA)
            if i % 100 == 0:
                print('------', curr_loss)
                if curr_loss > prev_loss_1k:
                    et = et * 0.1
                    print('lr/10 to', et)

                prev_loss_1k = curr_loss

            # early stopping
            if abs(curr_loss - prev_loss) < 1e-6:
                break
            prev_loss = curr_loss
        return W

    @staticmethod
    def _initialization(dim):
        """ Initialization.
        dim consists of (d1, d2,...), where dl = 1 (placeholder, deprecated). """
        w = []
        total_dim = 0
        for i in range(0, len(dim) - 1):
            if i < len(dim) - 2:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i + 1])
                temp = np.kron(np.eye(2, dtype=int), temp)
                temp = torch.from_numpy(temp)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i] * 4
            else:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i])
                temp = np.kron([[1, -1]], temp)
                temp = torch.from_numpy(temp)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i] * 2

        return w, total_dim

    @staticmethod
    def feature_extractor(x, W):
        """ Functions feature extractor.
        x is the input, dimension is d; W is the list of parameter matrices. """
        depth = len(W)
        output = x
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            output = output.clamp(min=0)

        output = output * math.sqrt(W[depth - 1].size()[1])
        return output

    def _gradient_loss(self, X, Y, W, THETA):
        """ Return a list of grad, satisfying that W[i] = W[i] - grad[i] for single context x. """
        depth = len(W)
        num_sample = Y.shape[0]
        loss = []
        grad = []
        relu = []
        output = X
        loss.append(output)
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            relu.append(output)
            output = output.clamp(min=0)
            loss.append(output)

        THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
        output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
        output = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)

        loss.append(output)

        feat = self.feature_extractor(X, W)
        feat_t = torch.transpose(feat, 0, 1).view(num_sample, -1, 1)
        output_t = torch.bmm(THETA_t, feat_t).squeeze().view(1, -1)

        # backward gradient propagation
        back = output_t - Y
        back = back.double()
        grad_t = torch.mm(back, loss[depth - 1].t())
        grad.append(grad_t)

        for i in range(1, depth):
            back = torch.mm(W[depth - i].t(), back)
            back[relu[depth - i - 1] < 0] = 0
            grad_t = torch.mm(back, loss[depth - i - 1].t())
            grad.append(grad_t)

        grad1 = []
        for i in range(0, depth):
            grad1.append(grad[depth - 1 - i] * math.sqrt(W[depth - 1].size()[1]) / len(X[0, :]))

        return grad1

    def _loss(self, X, Y, W, THETA):
        # total loss
        num_sample = len(X[0, :])
        output = self.feature_extractor(X, W)
        THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
        output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
        output_y = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)

        summ = (Y - output_y).pow(2).sum() / num_sample
        return summ
