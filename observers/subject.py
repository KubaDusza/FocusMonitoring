from abc import ABC, abstractmethod

from observers.observer import Observer


class Subject(ABC):

    def __init__(self):
        # Observers
        self.observers: [Observer] = []

    def add_observer(self, observer: Observer):
        """
        Add an observer to the list.
        """
        self.observers.append(observer)

    def remove_observer(self, observer):
        """
        Remove an observer from the list.
        """
        self.observers.remove(observer)

    def notify_observers(self, data):
        """
        Notify all observers with the given data.
        """
        for observer in self.observers:
            observer.update(data)