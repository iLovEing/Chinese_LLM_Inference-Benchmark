import os
import json
import pandas as pd

from ._base import ChoiceBenchmark, BaseChoiceBenchmark
from util import GlobalConfig


class CEval(BaseChoiceBenchmark):
    """
    CEval 不提供测试集标签，这里用val作为测试集
    """
    benchmark: str = 'CEval'

    def __init__(self, cfg: GlobalConfig):
        super().__init__(cfg)

        self.subjects = sorted([f.split("_val.csv")[0] for f in os.listdir(os.path.join(self.data_root, "val"))])
        self.data_csv = {
            _subject: {
                'test': os.path.join(self.data_root, 'val', f'{_subject}_val.csv'),
                'dev': os.path.join(self.data_root, 'dev', f'{_subject}_dev.csv'),
            }

            for _subject in self.subjects
        }

        with open(os.path.join(self.data_root, 'subject_mapping.json'), 'r', encoding='utf-8') as file:
            json_mapping = json.load(file)
        self.name_en2zh = {
            _subject: _info[1]
            for _subject, _info in json_mapping.items()
        }

    def get_question_answer(self, df):
        df_dict = df.to_dict(orient='list')

        qst = []
        ans = df_dict['answer']
        for idx in range(len(df_dict['question'])):
            prompt = f'题目： {df_dict["question"][idx]}'
            for _c in self.choices:
                prompt += f'\n{_c}. {df_dict[_c][idx]}'

            qst.append(prompt)

        return qst, ans

    def get_subject_bhm(self, subject) -> ChoiceBenchmark:
        dev_df = pd.read_csv(self.data_csv[subject]['dev'], header=0, index_col=0)
        test_df = pd.read_csv(self.data_csv[subject]['test'], header=0, index_col=0)

        dev_qst, dev_ans = self.get_question_answer(dev_df)
        test_qst, test_ans = self.get_question_answer(test_df)

        return ChoiceBenchmark(benchmark=self.benchmark,
                               name_EN=subject,
                               name_CH=self.name_en2zh[subject],
                               choices=self.choices,
                               test_qst=test_qst,
                               test_ans=test_ans,
                               dev_qst=dev_qst,
                               dev_ans=dev_ans)

