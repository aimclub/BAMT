from bamt.parameter_estimators.parameters_estimator import ParametersEstimator


class MaximumLikelihoodEstimator(ParametersEstimator):
    def __init__(self, network):
        super().__init__(network)

    def estimate(self):
        pass
