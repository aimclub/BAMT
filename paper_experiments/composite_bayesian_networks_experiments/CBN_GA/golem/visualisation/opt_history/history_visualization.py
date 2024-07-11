from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Any

from golem.core.log import default_log
from golem.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper

if TYPE_CHECKING:
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


class HistoryVisualization(metaclass=ArgConstraintWrapper):
    """ Base class for creating visualizations of the optimization history.
    The only necessary method is 'visualize' - it must show or save the plot in any form after the call.

    One should refer the OptHistory instance as `self.history` to be able to connect one's visualization
    to `OptHistory.show()`. See the examples of connecting visualizations in the module `opt_viz.py`.

    It is good practice to be aware of constraints on your visualizations. You can either implement
    default constraints that will catch your kwarg across all the visualizations or define your single
    class specific constraints by assigning them to `constraint_checkers` class attribute.
    See `golem.core.visualisation.opt_history.arg_constraint_wrapper.py` for examples.
    """
    constraint_checkers = []  # Use this for class-specific constraint checkers.

    def __init__(self, history: 'OptHistory', visuals_params: Dict[str, Any] = None):
        self.visuals_params = visuals_params or {}
        self.history = history
        self.log = default_log(self)

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError()

    def get_predefined_value(self, param: str):
        return self.visuals_params.get(param)
