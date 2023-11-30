from typing import List, Optional, Tuple, TypedDict, Sequence


class ParamDict(TypedDict, total=False):
    init_edges: Optional[Sequence[str]]
    init_nodes: Optional[List[str]]
    remove_init_edges: bool
    white_list: Optional[Tuple[str, str]]
    bl_add: Optional[List[str]]
