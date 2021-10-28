class BaseNode(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type

        self.parents = []
        self.children = []

    def __repr__(self):
        return f"{self.name}"


class DiscreteNode(BaseNode):
    def __init__(self, name, type):
        super(DiscreteNode, self).__init__(name, type)


class ContinousNode(BaseNode):
    def __init__(self, name, type):
        super(ContinousNode, self).__init__(name=name, type=type)


class GaussianNode(ContinousNode):
    def __init__(self, name, type):
        super(GaussianNode, self).__init__(name, type)
