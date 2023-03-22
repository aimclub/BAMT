from typing import Dict, List, Any, Union, TypedDict, Optional
from numpy import ndarray


class DiscreteParams(TypedDict):
    cprob: Union[List[Union[list, Any]], Dict[str, list]]
    vals: List[str]


class MixtureGaussianParams(TypedDict):
    mean: List[float]
    coef: List[float]
    covars: List[float]


class GaussianParams(TypedDict):
    regressor: str
    regressor_obj: Optional[Union[str, bool, bytes]]
    variance: Union[ndarray, float]
    mean: Union[ndarray, float]
    serialization: str


class CondGaussParams(TypedDict):
    regressor: str
    regressor_obj: Optional[Union[str, bool, bytes]]
    variance: Union[ndarray, float]
    mean: Union[ndarray, float]
    serialization: str


class CondMixtureGaussParams(TypedDict):
    mean: Optional[List[float]]
    coef: List[float]
    covars: Optional[List[float]]


class LogitParams(TypedDict):
    classes: List[int]
    classifier: str
    classifier_obj: Optional[Union[str, bool, bytes]]
    serialization: str
