class BaseNode(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __repr__(self):
        return f"{self.name}"


class DiscreteNode(BaseNode):
    def __init__(self, name, type):
        super(DiscreteNode, self).__init__(name, type)
        self.probas_matrix = []

class GaussianNode(BaseNode):
    def __init__(self,name,type):
        super(GaussianNode, self).__init__(name, type)
        self.mean = 0
        self.variance = 1
