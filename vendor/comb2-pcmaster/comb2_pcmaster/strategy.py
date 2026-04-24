from abc import ABCMeta, abstractmethod


class StrategyBase(metaclass=ABCMeta):
    def __init__(self, strategy_config: dict, dataloader):
        self.config = strategy_config
        self.dataloader = dataloader

    @abstractmethod
    def generate_positions(self, signals, last_hold):
        raise NotImplementedError
