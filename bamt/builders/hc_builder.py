from typing import Dict, List, Optional, Tuple, Callable, Union

from pandas import DataFrame
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge

from bamt.builders.builders_base import ParamDict, BaseDefiner
from bamt.log import logger_builder
from bamt.redef_HC import hc as hc_method
from bamt.utils import GraphUtils as gru


class HillClimbDefiner(BaseDefiner):
    """
    Object to define structure and pass it into skeleton
    """

    def __init__(
        self,
        data: DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        scoring_function: Union[Tuple[str, Callable], Tuple[str]],
        regressor: Optional[object] = None,
    ):
        super().__init__(data, descriptor, scoring_function, regressor)
        self.optimizer = HillClimbSearch(data)

    def apply_K2(
        self,
        data: DataFrame,
        init_edges: Optional[List[Tuple[str, str]]],
        progress_bar: bool,
        remove_init_edges: bool,
        white_list: Optional[List[Tuple[str, str]]],
    ):
        """
        :param init_edges: list of tuples, a graph to start learning with
        :param remove_init_edges: allows changes in a model defined by user
        :param data: user's data
        :param progress_bar: verbose regime
        :param white_list: list of allowed edges
        """
        if not all([i in ["disc", "disc_num"] for i in gru.nodes_types(data).values()]):
            logger_builder.error(
                f"K2 deals only with discrete data. Continuous data: {[col for col, type in gru.nodes_types(data).items() if type not in ['disc', 'disc_num']]}"
            )
            return None

        if len(self.scoring_function) != 2:
            from pgmpy.estimators import K2

            scoring_function = K2
        else:
            scoring_function = self.scoring_function[1]

        expert_knowledge = ExpertKnowledge(
            forbidden_edges=self.black_list,
            required_edges=white_list
        )

        if not init_edges:
            best_model = self.optimizer.estimate(
                scoring_method=scoring_function(data),
                expert_knowledge=expert_knowledge,
                show_progress=progress_bar,
            )
        else:
            if remove_init_edges:
                startdag = DAG()
                nodes = [str(v) for v in self.vertices]
                startdag.add_nodes_from(nodes=nodes)
                startdag.add_edges_from(ebunch=init_edges)
                best_model = self.optimizer.estimate(
                    expert_knowledge=expert_knowledge,
                    start_dag=startdag,
                    show_progress=False,
                )
            else:
                best_model = self.optimizer.estimate(
                    expert_knowledge=expert_knowledge,
                    show_progress=False,
                )

        structure = [list(x) for x in list(best_model.edges())]
        self.skeleton["E"] = structure

    def apply_group1(
        self,
        data: DataFrame,
        progress_bar: bool,
        init_edges: Optional[List[Tuple[str, str]]],
        remove_init_edges: bool,
        white_list: Optional[List[Tuple[str, str]]],
    ):
        """
        This method implements the group of scoring functions.
        Group:
        "MI" - Mutual Information,
        "LL" - Log Likelihood,
        "BIC" - Bayesian Information Criteria,
        "AIC" - Akaike information Criteria.
        """
        column_name_dict = dict([(n.name, i) for i, n in enumerate(self.vertices)])
        blacklist_new = []
        for pair in self.black_list:
            blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
        if white_list:
            white_list_old = white_list[:]
            white_list = []
            for pair in white_list_old:
                white_list.append(
                    (column_name_dict[pair[0]], column_name_dict[pair[1]])
                )
        if init_edges:
            init_edges_old = init_edges[:]
            init_edges = []
            for pair in init_edges_old:
                init_edges.append(
                    (column_name_dict[pair[0]], column_name_dict[pair[1]])
                )

        bn = hc_method(
            data,
            metric=self.scoring_function[0],
            restriction=white_list,
            init_edges=init_edges,
            remove_geo_edges=remove_init_edges,
            black_list=blacklist_new,
            debug=progress_bar,
        )
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]["parents"]:
                structure.append(
                    [
                        list(column_name_dict.keys())[
                            list(column_name_dict.values()).index(pa)
                        ],
                        list(column_name_dict.keys())[
                            list(column_name_dict.values()).index(rv)
                        ],
                    ]
                )
        self.skeleton["E"] = structure


class HCStructureBuilder(HillClimbDefiner):
    """
    Final object with build method
    """

    def __init__(
        self,
        data: DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        scoring_function: Tuple[str, Callable],
        regressor: Optional[object],
        has_logit: bool,
        use_mixture: bool,
    ):
        """
        :param data: train data
        :param descriptor: map for data
        """

        super(HCStructureBuilder, self).__init__(
            descriptor=descriptor,
            data=data,
            scoring_function=scoring_function,
            regressor=regressor,
        )
        self.use_mixture = use_mixture
        self.has_logit = has_logit

    def build(
        self,
        data: DataFrame,
        progress_bar: bool,
        classifier: Optional[object],
        regressor: Optional[object],
        params: Optional[ParamDict] = None,
        **kwargs,
    ):
        if params:
            for param, value in params.items():
                self.params[param] = value

        init_nodes = self.params.pop("init_nodes")
        bl_add = self.params.pop("bl_add")

        # Level 1
        self.skeleton["V"] = self.vertices

        self.restrict(data, init_nodes, bl_add)
        if self.scoring_function[0] == "K2":
            self.apply_K2(data=data, progress_bar=progress_bar, **self.params)
        elif self.scoring_function[0] in ["MI", "LL", "BIC", "AIC"]:
            self.apply_group1(data=data, progress_bar=progress_bar, **self.params)

        # Level 2

        self.get_family()
        self.overwrite_vertex(
            has_logit=self.has_logit,
            use_mixture=self.use_mixture,
            classifier=classifier,
            regressor=regressor,
        )
