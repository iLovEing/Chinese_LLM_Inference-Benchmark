import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from ._base import BaseModel
from util import GlobalConfig

"""
qwen2.5 key special token
<|im_start|>, <|im_end|>: BOS(151644), EOS(151645), BOS was inserted at EVERY head of segment
<|endoftext|>: pad token (151643)
"""

PROMPT_ID = 2
PROMPT_TEMPLATES = [
    # 0. no template
    '{}',  # filled with format string

    # 1. english template 1
    '<|im_start|>system\n'
    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
    '<|im_start|>user\n'
    '{}<|im_end|>\n'  # filled with format string
    '<|im_start|>assistant\n',

    # 2. english template 2
    '<|im_start|>system\n'
    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 请用简体中文回答。<|im_end|>\n'
    '<|im_start|>user\n'
    '{}<|im_end|>\n'  # filled with format string
    '<|im_start|>assistant\n',

    # 3. chinese template
    '<|im_start|>system\n'
    '你是通义千问，由阿里云团队开发，你是一个很有帮助的AI助手，请用简体中文回答<|im_end|>\n'
    '<|im_start|>user\n'
    '{}<|im_end|>\n'  # filled with format string
    '<|im_start|>assistant\n',
]


class Qwen2_5(BaseModel):
    def __init__(self, cfg: GlobalConfig):
        super().__init__(cfg)

        print(f'---------- initialize model {self.cfg.model_name_or_path} ----------')
        print(f'##### loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.show_tokenizer()

        print('##### loading model...')
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True) if self.cfg.load_in_8bit else None
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.bfloat16,
                                                          device_map="auto", quantization_config=quant_cfg)
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
            generate_cfg = GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                num_beams=self.cfg.num_beams,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            result = self.model.generate(
                **model_input,
                generation_config=generate_cfg,
                # return_dict_in_generate=True
            )
            print(f'model inference finished, output shape: {result.shape}')
            output_text = [self.tokenizer.decode(_tokens, skip_special_tokens=self.cfg.skip_special_tokens)
                           for _tokens in result]

        return output_text

