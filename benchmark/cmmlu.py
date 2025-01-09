import os
import pandas as pd

from ._base import ChoiceBenchmark
from util import BenchmarkConfig


name_en2zh = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history":"中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science":"大学精算学",
    "college_education":"大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics":"大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese":"小学语文",
    "elementary_commonsense":"小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study":"安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history":"世界历史",
    "world_religions": "世界宗教",
}


class CMMLU:
    bhm_type: str = 'choice'

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg


        self.data_dir = self.cfg.data_dir
        self.save_dir = os.path.join(self.cfg.result_dir, self.cfg.model, self.cfg.benchmark)
        self.subjects = sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(cfg.data_dir, "test/"))])
        print(f'##### init CMMLU benchmark, data dir: {self.data_dir}, subjects num: {len(self.subjects)}\n')

    def get_choices(self, df: pd.DataFrame):
        return list(df.columns)[1: -1]

    def get_question_answer(self, df):
        choices = self.get_choices(df)
        df_dict = df.to_dict(orient='list')

        qst = []
        ans = df_dict['Answer']
        for idx in range(len(df_dict['Question'])):
            prompt = f'题目： {df_dict["Question"][idx]}'
            for _c in choices:
                prompt += f'\n{_c}. {df_dict[_c][idx]}'

            qst.append(prompt)

        return qst, ans

    def run_benchmark(self, model_api):
        print(f'----- run CMMLU benchmark, model {self.cfg.model} -----')
        os.makedirs(self.save_dir, exist_ok=True)

        for subject in self.subjects:
            dev_df = pd.read_csv(os.path.join(self.data_dir, "dev", subject + ".csv"), header=0, index_col=0)
            test_df = pd.read_csv(os.path.join(self.data_dir, "test", subject + ".csv"), header=0, index_col=0)

            choices = self.get_choices(test_df)
            dev_qst, dev_ans = self.get_question_answer(dev_df)
            test_qst, test_ans = self.get_question_answer(test_df)

            result_csv_name = f'{"un" if not self.cfg.strict_bhm else ""}strict_{self.cfg.few_shot}_shot_{subject}.csv'
            result_csv = os.path.join(self.save_dir, result_csv_name)
            if not self.cfg.force_refresh and os.path.exists(result_csv):  # If result file exist, skip this subject
                print(f'file {result_csv} already exists, skip refresh.')
                continue

            generate_results, choice_results, logits = model_api(
                ChoiceBenchmark(benchmark=self.cfg.benchmark,
                                name_EN=subject,
                                name_CH=name_en2zh[subject],
                                choices=choices,
                                test_qst=test_qst,
                                test_ans=test_ans,
                                dev_qst=dev_qst,
                                dev_ans=dev_ans)
            )

            assert len(test_qst) == len(generate_results) == len(choice_results) == len(logits)
            result_df = pd.DataFrame({
                'question': test_qst,
                'label': test_ans,
                'generate_result': generate_results,
                'choice_result': choice_results,
                'logits': logits,
            })
            result_df.to_csv(result_csv, encoding='utf-8', index=False)

        self.summary()
        print(f'---------- run CMMLU benchmark finish. ----------')

    def summary(self):
        summary_file_name = f'_{"un" if not self.cfg.strict_bhm else ""}strict_{self.cfg.few_shot}_shot_summary.txt'
        summary = os.path.join(self.save_dir, summary_file_name)
        if not self.cfg.force_refresh and os.path.exists(summary):
            print(f'file {summary} already exists, skip refresh.')
            return

        subject_result_csv = os.listdir(self.save_dir)
        subject_result_csv = list(filter(lambda f: f.endswith('.csv'), subject_result_csv))

        total_acc = 0.
        total_items = 0
        summary_str = ''
        for csv in sorted(subject_result_csv):
            subject_name = csv.split('.')[-2].split('result_')[-1]
            csv_f = os.path.join(self.save_dir, csv)
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
                    f'model: [{self.cfg.model}]-[{self.cfg.model_name_or_path}]')
        print(temp_str)
        summary_str = temp_str + '\n\n' + summary_str
        with open(summary, 'w', encoding='utf-8') as f:
            f.write(summary_str)

