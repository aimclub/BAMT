

class BaseSAApproachResult:
    """ Base class for presenting all result classes.
    Specifies the main logic of setting and getting calculated metrics. """

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns the worst result among all metrics. """
        raise NotImplementedError()

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        """ Returns the worst result with additional info. """
        raise NotImplementedError()

    def add_results(self, **kwargs):
        """ Adds newly calculated results. """
        raise NotImplementedError

    def get_dict_results(self) -> dict:
        """ Returns dict representation of results. """
        raise NotImplementedError()
