import os
import argparse
import importlib

from util import GlobalConfig


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yml")
    args = parser.parse_args()

    # read config
    g_config = GlobalConfig(cfg_path=os.path.join('config', args.config))

    # get model
    model_module = getattr(importlib.import_module('model'), g_config.model_module)  # dynamic load model
    model = model_module(cfg=g_config)

    if g_config.do_test_infer:
        model.run_generate()
    if g_config.do_benchmark:
        for benchmark in g_config.bhm_tasks:
            # get benchmark
            benchmark_module = getattr(importlib.import_module('benchmark'), benchmark)
            benchmark = benchmark_module(cfg=g_config)
            benchmark.run_benchmark(model.choice_bhm_api)


if __name__ == '__main__':
    main()

