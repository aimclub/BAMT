from typing import Optional

from sklearn import linear_model

from .logit_node import LogitNode


class CompositeDiscreteNode(LogitNode):
    """
    Class for composite discrete node.
    """

    def __init__(self, name, classifier: Optional[object] = None):
        super(CompositeDiscreteNode, self).__init__(name)
        if classifier is None:
            classifier = linear_model.LogisticRegression(
                solver="newton-cg", max_iter=100
            )
        self.classifier = classifier
        self.type = "CompositeDiscrete" + f" ({type(self.classifier).__name__})"
