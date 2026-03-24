from abc import ABC, abstractmethod


class Backend(ABC):

    name = "base"

    @abstractmethod
    def array(self, data):
        pass

    @abstractmethod
    def zeros(self, shape):
        pass

    @abstractmethod
    def ones(self, shape):
        pass

    @abstractmethod
    def matmul(self, a, b):
        pass

    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def sub(self, a, b):
        pass

    @abstractmethod
    def mul(self, a, b):
        pass

    @abstractmethod
    def div(self, a, b):
        pass

    @abstractmethod
    def sum(self, a, axis=None):
        pass

    @abstractmethod
    def mean(self, a, axis=None):
        pass

    @abstractmethod
    def exp(self, a):
        pass

    @abstractmethod
    def log(self, a):
        pass

    @abstractmethod
    def reshape(self, a, shape):
        pass

    @abstractmethod
    def transpose(self, a):
        pass