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

    @abstractmethod
    def choice_bhm_api(self, subject: ChoiceBenchmark):
        pass

