from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, data):
        """
        React to updates from the subject.

        Args:
            data: The data provided by the subject (e.g., rotation matrix, zone status).
        """
        raise NotImplementedError
