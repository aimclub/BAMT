from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass
