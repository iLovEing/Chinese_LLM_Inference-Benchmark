import os
import argparse
import importlib

from util import BenchmarkConfig


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="base")
    args = parser.parse_args()

    # read config
    benchmark_cfg = BenchmarkConfig(cfg_path=os.path.join('config', f'{args.config_name}.yml'))\

    # get benchmark
    benchmark_module = getattr(importlib.import_module('benchmark'), benchmark_cfg.benchmark)
    benchmark = benchmark_module(cfg=benchmark_cfg)

    # get model
    model_module = getattr(importlib.import_module('model'), benchmark_cfg.model)  # dynamic load model
    model = model_module(cfg=benchmark_cfg)

    if benchmark_cfg.do_test_infer:
        model.run_generate()
    if benchmark_cfg.do_benchmark:
        benchmark.run_benchmark(model.choice_bhm_api)


if __name__ == '__main__':
    main()

