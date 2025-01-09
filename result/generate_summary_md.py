import os
import re
import pandas as pd

def get_bhm_score(model, bhm):
    def _parse_cmmlu_score(_txt):
        with open(_txt, 'r', encoding='utf-8') as f:
            fst_line = f.readline()
            score_index = fst_line.find('acc') + len('acc')
            score = re.findall(r'\d+\.\d+|\d+', fst_line[score_index:])[0]
            return f'{float(score) * 100.0:.2f}'


    score = None
    if bhm == "CMMLU":
        txt_file = os.path.join(model, bhm, '_unstrict_5_shot_summary.txt')
        if os.path.exists(txt_file):
            score = _parse_cmmlu_score(txt_file)

    return score


def parse_bhm_summary():
    summary = {}
    benchmarks = []

    for model in os.listdir("."):
        if not os.path.isdir(model):
            continue

        summary[model] = {}
        for bhm in os.listdir(model):
            if not os.path.isdir(os.path.join(model, bhm)):
                continue
            benchmarks.append(bhm) if bhm not in benchmarks else None
            summary[model][bhm] = get_bhm_score(model, bhm)

    summary_df = pd.DataFrame(data=None, columns=['model'] + benchmarks)
    for model_name, bhm_info in summary.items():
        summary_df.loc[len(summary_df.index)] = [model_name] + [
            (bhm_info[_bhm] if _bhm in bhm_info.keys() else 'Nan') for _bhm in benchmarks
        ]

    summary_df.set_index('model', inplace=True)
    print(f'parse summary:\n{summary_df}')
    return summary_df


def generate_bhm_md():
    md_str = '# Benchmark Summary'
    summary_df = parse_bhm_summary()
    models = summary_df.index.tolist()
    benchmarks = summary_df.columns.tolist()

    gl = 15  # grid len
    md_str += f'\n|{"model":^{gl}}|'
    for bhm in benchmarks:
        md_str += f'{bhm:^{gl}}|'

    md_str += '\n|'
    for _ in range(len(benchmarks) + 1):
        md_str += (' ' + '-' * (gl - 2) + ' |')

    for model in models:
        md_str += f'\n|{model:^{gl}}|'
        for score in summary_df.loc[model].tolist():
            md_str += f'{score:^{gl}}|'

    print(md_str)
    with open('bhm_summary.md', 'w', encoding='utf-8') as f:
        f.write(md_str)


def generate_infer_md():
    md_str = '# Inference Summary'
    for model in os.listdir("."):
        if not os.path.isdir(model):
            continue
        infer_result = os.path.join(model, 'infer_result.md')
        if not os.path.isfile(infer_result):
            continue

        with open(infer_result, 'r', encoding='utf-8') as f:
            content = f.read()
            md_str += ('\n\n' + content)

    print(md_str)
    with open('infer_summary.md', 'w', encoding='utf-8') as f:
        f.write(md_str)

def main():
    generate_bhm_md()
    generate_infer_md()


if __name__ == '__main__':
    main()

