import os
import pandas as pd
from benchmark import ChoiceBenchmark
from util import BenchmarkConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from abc import ABC, abstractmethod


# base benchmark pipeline
class BaseBHM(ABC):
    cfg: BenchmarkConfig
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    prompt_templates: str

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg

    def show_tokenizer(self):
        print(self.tokenizer.vocab_size, self.tokenizer.model_max_length)
        print(self.tokenizer.special_tokens_map)
        print(self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
        print(self.tokenizer.added_tokens_encoder)

        test_str = '你好。'
        print(f'##### tokenizer test str: {test_str}')
        test_encode = self.tokenizer.encode('你好。', return_tensors='pt', add_special_tokens=False)
        test_decode = self.tokenizer.decode(test_encode[0])
        print(f'encode: {test_encode}')
        print(f'decode: {test_decode}')

    def show_model(self):
        print(f'##### model device: {self.model.device}')
        print(self.model.config)

    def generate_prompt(self, input_txt: list[str]):
        return self.prompt_templates.format(*input_txt)

    @abstractmethod
    def run_generate(self):
        pass

    def run_benchmark(self, bhm):
        print(f'---------- run benchmark, model: {self.cfg.model_name_or_path}, bhm: {self.cfg.benchmark} ----------')
        if bhm.bhm_type == 'choice':
            os.makedirs(self.cfg.result_dir, exist_ok=True)

            for subject in bhm.get_subject():
                self.run_choice_benchmark(subject)

            self.summary_choice_bhm()
        print(f'---------- run benchmark finish. ----------')

    @abstractmethod
    def run_choice_benchmark(self, subject: ChoiceBenchmark):
        pass

    def summary_choice_bhm(self):
        summary = os.path.join(self.cfg.result_dir, '0.summary.txt')
        if not self.cfg.force_refresh and os.path.exists(summary):
            return

        subject_result_csv = os.listdir(self.cfg.result_dir)
        subject_result_csv = list(filter(lambda f: f.endswith('.csv'), subject_result_csv))

        total_acc = 0.
        total_items = 0
        summary_str = ''
        for csv in sorted(subject_result_csv):
            subject_name = csv.split('.')[-2].split('result_')[-1]
            csv_f = os.path.join(self.cfg.result_dir, csv)
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

        temp_str = f'summary: num {total_items} | acc {total_acc / total_items:.5f}'
        print(temp_str)
        summary_str = temp_str + '\n\n' + summary_str
        with open(summary, 'w', encoding='utf-8') as f:
            f.write(summary_str)


