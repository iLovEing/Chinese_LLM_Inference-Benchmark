import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from .model_base import BaseBHM
from util import BenchmarkConfig
from benchmark import ChoiceBenchmark


PROMPT_ID = 1
PROMPT_TEMPLATES = [
    # 0. no template
    '<|begin_of_text|>'
    '{}',  # filled with format string

    # 1. english template 1
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
    'You are a helpful AI assistant.'
    '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
    '{}'  # filled with format string
    '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',

    # 2. english template 2
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
    'You are a helpful AI assistant, 请用简体中文回答.'
    '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
    '{}'  # filled with format string
    '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',

    # 3. chinese template
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
    '你是一个很有帮助的AI助手，请用简体中文回答.'
    '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
    '{}'  # filled with format string
    '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
]


class Llama3(BaseBHM):
    def __init__(self, cfg: BenchmarkConfig):
        super().__init__(cfg)

        print(f'---------- initialize model {self.cfg.model_name_or_path} ----------')
        print(f'##### loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.show_tokenizer()

        print('##### loading model...')
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.show_model()

        self.prompt_templates = PROMPT_TEMPLATES[PROMPT_ID]

        print(f'---------- initialize model finished. ----------\n')

    def run_generate(self):
        print(f'---------- run generate, model {self.cfg.model_name_or_path} ----------')
        prompts = [self.generate_prompt([_input]) for _input in self.cfg.generate_input]
        model_input = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
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
            print(f'##### output shape: {result.shape}')
            for idx in range(result.shape[0]):
                print(f'\n##### case {idx + 1}:')
                print(self.tokenizer.decode(result[idx], skip_special_tokens=self.cfg.skip_special_tokens))

        print(f'---------- run generate finish. ----------')

    def generate_choice_bhm_prompt(self, bhm_subject: ChoiceBenchmark):
        """
        generate prompt according to few_shot and max_length
        """
        name = bhm_subject.name_CH
        test_qst = bhm_subject.test_qst
        dev_qst = bhm_subject.dev_qst
        dev_ans = bhm_subject.dev_ans
        prompts = []
        few_shot_ids = list(range(len(dev_qst)))

        for idx in range(len(test_qst)):
            if self.cfg.random_shot:
                random.shuffle(few_shot_ids)
            question = test_qst[idx]
            input_head = f'以下是关于{name}的单项选择题，请直接给出正确答案的选项。\n\n'

            few_shot_count = min(self.cfg.few_shot, len(few_shot_ids))
            few_shot_id = 0
            few_shot_input = ''
            final_input = ''
            while few_shot_count >= 0:
                if few_shot_id < len(few_shot_ids):
                    idx = few_shot_ids[few_shot_id]
                    single_shot_input = dev_qst[idx] + '\n答案是：' + dev_ans[idx] + '\n\n'
                else:
                    single_shot_input = ''

                temp_input = input_head + few_shot_input + question + '\n答案是：'
                temp_tokenized = self.tokenizer.encode(self.generate_prompt([final_input]),
                                                       return_tensors='pt', add_special_tokens=False)
                if temp_tokenized.shape[1] > self.cfg.max_length:
                    if few_shot_id == 0:
                        print(f'warning too long input: {question}')
                        final_input = temp_input
                    break

                final_input = temp_input
                few_shot_input += single_shot_input
                few_shot_count -= 1
                few_shot_id += 1

            prompts.append(final_input)
        return prompts

    def choice_bhm_api(self, bhm_subject: ChoiceBenchmark):
        assert self.cfg.few_shot >= 0, f'invalid arg few_shot: {self.cfg.few_shot}'

        prompts = self.generate_choice_bhm_prompt(bhm_subject)
        infer_loader = DataLoader(prompts,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: self.tokenizer(x, padding=True, return_tensors="pt", add_special_tokens=False))

        choice_tokenizer_idx = []
        for _choice in bhm_subject.choices:
            choice_tokenizer_idx += self.tokenizer.encode(_choice, add_special_tokens=False)
        generate_results = []
        choice_results = []
        logits = []
        with torch.no_grad():
            for batch in tqdm(infer_loader, desc=f'run choice benchmark [{self.cfg.benchmark}-{bhm_subject.name_EN}-{bhm_subject.name_CH}]'):
                for _k, _ in batch.items():
                    batch[_k] = batch[_k].to(self.model.device)
                model_output = self.model(**batch)

                generate_logits = model_output['logits'][:, -1, :].detach().cpu()
                generate_result = generate_logits.argmax(-1)
                generate_result = [self.tokenizer.decode(generate_result[_idx]) for _idx in range(generate_result.shape[0])]
                generate_results += generate_result

                choice_logits = generate_logits[:, choice_tokenizer_idx]
                choice_result = [bhm_subject.choices[i] for i in choice_logits.argmax(-1).tolist()]
                choice_results += choice_result
                for _idx in range(choice_logits.shape[0]):
                    logits.append(','.join(map(str, choice_logits[_idx].tolist())))

        return generate_results, choice_results, logits

