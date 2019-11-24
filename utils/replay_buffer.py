import numpy as np
from abc import ABC, abstractmethod
import json

# [s, visual_s, a, r, s_, visual_s_, done] must be this format.


class ReplayBuffer(ABC):
    def __init__(self, batch_size, capacity):
        assert isinstance(batch_size, int) and batch_size > 0, 'batch_size must be int and larger than 0'
        assert isinstance(capacity, int) and capacity > 0, 'capacity must be int and larger than 0'
        self.batch_size = batch_size
        self.capacity = capacity
        self._size = 0

    @abstractmethod
    def sample(self) -> list:
        pass

    def is_empty(self):
        return self._size == 0

    def update(self, *args) -> None:
        pass


class ExperienceReplay(ReplayBuffer):
    def __init__(self, batch_size, capacity):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, data):
        self._store_op(data)

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def sample(self):
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.array(e) for e in zip(*t)]

    def get_all(self):
        return [np.array(e) for e in zip(*self._buffer[:self._size])]

    def update_rb_after_add(self):
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    @property
    def show_rb(self):
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:, np.newaxis])
