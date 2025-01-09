import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ._base import BaseBHM
from util import BenchmarkConfig


"""
blueLM key special token
<s>, </s>: BOS(1), EOS(2)
<pad>: pad token(3)
<[|Human|]>, [|AI|]: role token(100000, 100001)
"""

PROMPT_ID = 1
PROMPT_TEMPLATES = [
    # 0. no template
    '{}',  # filled with format string

    # 1. english template 1
    '[|Human|]:'
    '{}'
    '\n[|AI|]:',
]


class BlueLM(BaseBHM):
    def __init__(self, cfg: BenchmarkConfig):
        super().__init__(cfg)

        print(f'---------- initialize model {self.cfg.model_name_or_path} ----------')
        print(f'##### loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True, use_fast=False,
                                                       add_bos_token=True, add_eos_token=False)
        self.tokenizer.padding_side = 'left'
        self.show_tokenizer()

        print('##### loading model...')
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True) if self.cfg.load_in_8bit else None
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True, device_map="auto",
                                                          quantization_config=quant_cfg)
        self.model.eval()
        self.show_model()

        self.prompt_templates = PROMPT_TEMPLATES[PROMPT_ID]

        print(f'---------- initialize model finished. ----------\n')

    def generate_text(self, input_text: list[str]) -> list[str]:
        model_input = self.tokenizer(input_text, return_tensors='pt', padding=True)
        for _k, _ in model_input.items():
            model_input[_k] = model_input[_k].to(self.model.device)

        print('run model inference...')
        with torch.no_grad():
            result = self.model.generate(
                **model_input,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                num_beams=self.cfg.num_beams,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                # return_dict_in_generate=True
            )
            print(f'model inference finished, output shape: {result.shape}')
            output_text = [self.tokenizer.decode(_tokens, skip_special_tokens=self.cfg.skip_special_tokens)
                           for _tokens in result]

        return output_text

