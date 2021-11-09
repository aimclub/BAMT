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


class GaussianNode(BaseNode):
    def __init__(self, name):
        super(GaussianNode, self).__init__(name)
        self.type = 'Gaussian'


class ConditionalGaussianNode(BaseNode):
    def __init__(self, name):
        super(ConditionalGaussianNode, self).__init__(name)
        self.type = 'ConditionalGaussian'


class MixtureGaussianNode(BaseNode):
    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = 'MixtureGaussian'


class ConditionalMixtureGaussianNode(BaseNode):
    def __init__(self, name):
        super(ConditionalMixtureGaussianNode, self).__init__(name)
        self.type = 'ConditionalMixtureGaussian'


class LogitNode(DiscreteNode):
    def __init__(self, name):
        super(LogitNode, self).__init__(name)
        self.type = 'Logit'


class ConditionalLogitNode(DiscreteNode):
    def __init__(self, name):
        super(ConditionalLogitNode, self).__init__(name)
        self.type = 'ConditionalLogit'
