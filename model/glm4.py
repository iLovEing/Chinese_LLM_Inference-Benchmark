import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from ._base import BaseModel
from util import GlobalConfig

"""
GLM4 key special token
[gMASK]: 151331, 
<|system|>, <|user|>, <|assistant|>: 151335, 151336, 151337, role token
<|endoftext|>: 151329, EOS and PAD
<sop>: 151333, BOS

in chat, role token will be treated as EOS
"""

PROMPT_ID = 2
PROMPT_TEMPLATES = [
    # 0. no template
    '{}',  # filled with format string

    # 1. chat
    '<|user|>\n'
    '{}'
    '<|assistant|>\n',

    # 2. instruct with system
    '<|system|>\n'
    '你是ChatGLM, 一个非常有用的人工智能助手. 请用简体中文回答。<|user|>\n'
    '{}'
    '<|assistant|>\n',
]


class GLM4(BaseModel):
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
                                                          device_map="auto", trust_remote_code=True,
                                                          quantization_config=quant_cfg)
        self.model.eval()
        self.show_model()

        self.prompt_templates = PROMPT_TEMPLATES[PROMPT_ID]

        print(f'---------- initialize model finished. ----------\n')

    def generate_text(self, input_text: list[str]) -> list[str]:
        model_input = self.tokenizer(input_text, return_tensors='pt', padding=True)
        model_input = model_input.to(self.model.device)

        print('run model inference...')
        with torch.no_grad():
            generate_cfg = GenerationConfig(
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

