import os
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from util import GlobalConfig
from benchmark import ChoiceBenchmark


# base benchmark pipeline
class BaseModel(ABC):
    cfg: GlobalConfig
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    prompt_templates: str

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.save_root = os.path.join(self.cfg.result_dir, self.cfg.model)
        os.makedirs(self.save_root, exist_ok=True)

    def show_tokenizer(self):
        print(self.tokenizer.vocab_size, self.tokenizer.model_max_length)
        print(self.tokenizer.special_tokens_map)
        print(self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
        print(self.tokenizer.added_tokens_encoder)

        test_str = '你好。'
        print(f'##### tokenizer test str: {test_str}')
        test_encode = self.tokenizer.encode('你好。', return_tensors='pt')
        test_decode = self.tokenizer.decode(test_encode[0])
        print(f'encode: {test_encode}')
        print(f'decode: {test_decode}')

    def show_model(self):
        print(f'##### model device: {self.model.device}')
        print(self.model.config)

    def generate_prompt(self, input_txt: list[str]):
        return self.prompt_templates.format(*input_txt)

    @abstractmethod
    def generate_text(self, input_txt: list[str]) -> list[str]:
        return []

    def run_generate(self):
        print(f'---------- run generate, model {self.cfg.model_name_or_path} ----------')
        prompts = [self.generate_prompt([_input]) for _input in self.cfg.generate_input]
        output_text = self.generate_text(prompts)

        result_str = f'## model: [{self.cfg.model}]-[{self.cfg.model_name_or_path}]'
        for idx in range(len(output_text)):
            result_str += f'\n\n### case {idx + 1}'
            result_str += ('\n```\n' + output_text[idx] + '\n```')

        print(result_str)
        save_f = os.path.join(self.save_root, 'infer_result.md')
        if not os.path.exists(save_f) or self.cfg.force_refresh:
            with open(save_f, 'w', encoding='utf-8') as f:
                f.write(result_str)
        else:
            print(f'\nfile {save_f} already exists, skip refresh.')
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
                temp_tokenized = self.tokenizer.encode(self.generate_prompt([final_input]), return_tensors='pt')
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

    def choice_bhm_api(self, benchmark: str, bhm_subject: ChoiceBenchmark):
        assert self.cfg.few_shot >= 0, f'invalid arg few_shot: {self.cfg.few_shot}'

        prompts = self.generate_choice_bhm_prompt(bhm_subject)
        infer_loader = DataLoader(prompts,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: self.tokenizer(x, padding=True, return_tensors="pt"))

        choice_tokenizer_idx = []
        for _choice in bhm_subject.choices:
            choice_tokenizer_idx += self.tokenizer.encode(_choice, add_special_tokens=False)
        generate_results = []
        choice_results = []
        logits = []
        with torch.no_grad():
            for batch in tqdm(infer_loader, desc=f'run choice benchmark [{benchmark}-{bhm_subject.name_EN}-{bhm_subject.name_CH}]'):
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

