from .cpu_backend import CPUBackend


class BackendManager:

    _backend = CPUBackend()

    @classmethod
    def set_device(cls, device):

        if device != "cpu":
            print("Only CPU backend available. Using CPU.")

        cls._backend = CPUBackend()

    @classmethod
    def get_backend(cls):
        return cls._backend