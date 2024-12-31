import random
import numpy as np

from .model_base import BaseBHM
from util import BenchmarkConfig
from benchmark import ChoiceBenchmark


class Random(BaseBHM):
    def __init__(self, cfg: BenchmarkConfig, seed:int =None):
        super().__init__(cfg)
        self.seed = seed

    def run_generate(self):
        pass

    def choice_bhm_api(self, bhm_subject: ChoiceBenchmark):
        questions, labels = bhm_subject.test_qst, bhm_subject.test_ans
        choices = bhm_subject.choices
        item_num = len(questions)
        choices_num = len(choices)
        results = []
        logits = []

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        rands = np.random.rand(item_num, choices_num)
        for idx in range(rands.shape[0]):
            results.append(choices[np.argmax(rands[idx])])
            logits.append(','.join(map(str, rands[idx].tolist())))

        return results, results, logits

