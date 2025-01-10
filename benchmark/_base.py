import os
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

from util import GlobalConfig


@dataclass
class ChoiceBenchmark:
    benchmark: str = None
    name_EN: str = None
    name_CH: str = None

    choices: list[str] = ()
    test_qst: list[str] = ()
    test_ans: list[str] = ()
    dev_qst: list[str] = ()
    dev_ans: list[str] = ()

    def __init__(self,
                 benchmark: str = None,
                 name_EN: str = None,
                 name_CH: str = None,
                 choices: list[str] = (),
                 test_qst: list[str] = (),
                 test_ans: list[str] = (),
                 dev_qst: list[str] = (),
                 dev_ans: list[str] = (),
                 ):
        self.benchmark = benchmark
        self.name_EN = name_EN
        self.name_CH = name_CH
        self.choices = choices
        self.test_qst = test_qst
        self.test_ans = test_ans
        self.dev_qst = dev_qst
        self.dev_ans = dev_ans

        assert len(self.choices) > 1, f'invalid choice: {self.choices}'
        assert 0 < len(self.test_qst) == len(self.test_ans), f'invalid test case len: {len(self.test_qst)}, {len(self.test_ans)}'
        assert len(self.dev_ans) == len(self.dev_qst), f'invalid dev case len: {len(self.dev_qst)}, {len(self.dev_ans)}'

    def __repr__(self):
        result = ''
        result += f'{self.__class__.__name__}\n'
        for k, v in self.__dict__.items():
            result += f'{k}: {v}\n'

        return result.strip()

class BaseChoiceBenchmark(ABC):
    benchmark: str = None
    bhm_type: str = 'choice'

    cfg: GlobalConfig = None
    save_root: str = None
    data_root: str = None

    subjects: list[str] = ()
    choices: list[str] = ['A', 'B', 'C', 'D']

    def __init__(self, cfg: GlobalConfig):
        print(f'##### init {self.__class__.__name__} benchmark\n')
        self.cfg = cfg
        self.save_root = os.path.join(cfg.result_dir, cfg.model, self.benchmark)
        self.data_root = os.path.join(cfg.data_dir, self.benchmark)

    @abstractmethod
    def get_subject_bhm(self, subject) -> ChoiceBenchmark:
        pass

    def run_benchmark(self, model_api):
        assert self.benchmark is not None, f'invalid class without benchmark name.'

        print(f'----- run {self.__class__.__name__} benchmark, model {self.cfg.model} -----')
        os.makedirs(self.save_root, exist_ok=True)

        for subject in self.subjects:
            result_csv_name = f'{"un" if not self.cfg.strict_bhm else ""}strict_{self.cfg.few_shot}_shot_{subject}.csv'
            result_csv = os.path.join(self.save_root, result_csv_name)
            if not self.cfg.force_refresh and os.path.exists(result_csv):  # If result file exist, skip this subject
                print(f'file {result_csv} already exists, skip refresh.')
                continue

            subject_bhm = self.get_subject_bhm(subject)
            generate_results, choice_results, logits = model_api(self.benchmark, subject_bhm)

            result_df = pd.DataFrame({
                'question': subject_bhm.test_qst,
                'label': subject_bhm.test_ans,
                'generate_result': generate_results,
                'choice_result': choice_results,
                'logits': logits,
            })
            result_df.to_csv(result_csv, encoding='utf-8', index=False)

        self.summary()
        print(f'---------- run {self.__class__.__name__} benchmark finish. ----------')

    def summary(self):
        summary_file_name = f'_{"un" if not self.cfg.strict_bhm else ""}strict_{self.cfg.few_shot}_shot_summary.txt'
        summary = os.path.join(self.save_root, summary_file_name)
        if not self.cfg.force_refresh and os.path.exists(summary):
            print(f'file {summary} already exists, skip refresh.')
            return

        subject_result_csv = os.listdir(self.save_root)
        subject_result_csv = list(filter(lambda f: f.endswith('.csv'), subject_result_csv))

        total_acc = 0.
        total_items = 0
        summary_str = ''
        for csv in sorted(subject_result_csv):
            subject_name = csv.split('.')[-2].split('result_')[-1]
            csv_f = os.path.join(self.save_root, csv)
            subject_df = pd.read_csv(csv_f, encoding='utf-8', index_col=None)

            if self.cfg.strict_bhm:
                subject_acc = (subject_df['label'] == subject_df['generate_result']).sum()
            else:
                subject_acc = (subject_df['label'] == subject_df['choice_result']).sum()
            subject_item_count = len(subject_df)

            total_acc += subject_acc
            total_items += subject_item_count

            temp_str = f'subject: {subject_name:<25} | num: {subject_item_count:<5} | ' \
                       f'acc: {subject_acc / subject_item_count:.5f}'
            print(temp_str)
            summary_str += f'{temp_str}\n'

        temp_str = (f'summary: total num {total_items} | acc {total_acc / total_items:.5f}\n'
                    f'model: [{self.cfg.model}]')
        print(temp_str)
        summary_str = temp_str + '\n\n' + summary_str
        with open(summary, 'w', encoding='utf-8') as f:
            f.write(summary_str)



