
import os
import random
import pandas as pd
import numpy as np

from .model_base import BaseBHM
from util import BenchmarkConfig
from benchmark import ChoiceBenchmark


class Random(BaseBHM):
    def __init__(self, cfg: BenchmarkConfig, seed:int =None):
        super().__init__(cfg)
        self.seed = seed

    def run_test_infer(self):
        pass

    def run_choice_benchmark(self, subject: ChoiceBenchmark):
        result_csv = os.path.join(self.cfg.result_dir, f"result_{subject.name_EN}.csv")
        if not self.cfg.force_refresh and os.path.exists(result_csv):  # If result file exist, skip this subject
            return

        questions, labels = subject.test_qst, subject.test_ans
        choices = subject.choices
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

        result_df = pd.DataFrame({
            'question': questions,
            'label': labels,
            'generate_result': results,
            'choice_result': results,
            'logits': logits,
        })
        result_df.to_csv(result_csv, encoding='utf-8', index=False)