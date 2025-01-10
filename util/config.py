import yaml
import platform
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    # paths
    cfg_path: str = None
    model_name_or_path: str = None
    data_dir: str = None
    result_dir: str = None


    # behavior
    bhm_tasks: list[str] = ()  # benchmark name and module
    model: str = None  # model name
    model_module: str = None  # model class name
    load_in_8bit: bool = False
    do_test_infer: bool = False
    do_benchmark: bool = False
    force_refresh: bool = False  # force refresh result

    # generate_config
    skip_special_tokens: bool = False  # set true to avoid show special_tokens such as pad_token
    max_new_tokens: int = 1024
    do_sample: bool = True
    num_beams: int = 5
    temperature: float = 0.9
    top_p: float = 0.75
    top_k: float = 50
    generate_input: list[str] = ()  # test infer input

    # benchmark config
    few_shot: int = 5  # set 0 means zero-shot
    random_shot: bool = False
    max_length: int = 1024
    batch_size: int = 1
    strict_bhm: bool = False  # if


    def __init__(self, cfg_path):
        print(f'---------- read config {cfg_path} ----------')
        if cfg_path is not None:
            self.cfg_path = cfg_path
            self.parse_config()
        print(self)
        print(f'---------- read config finished. ----------\n')

    def __repr__(self):
        result = ''
        result += f'{self.__class__.__name__}\n'
        for k, v in self.__dict__.items():
            result += f'{k}: {v}\n'

        return result.strip()

    def pre_adjust_config(self, yaml_cfg):
        # sys_compatible
        sys_key = 'win' if platform.system() == 'Windows' else 'linux'
        if 'paths' in yaml_cfg:
            for _k, _v in yaml_cfg['paths'].items():
                yaml_cfg[_k] = yaml_cfg['paths'][_k][sys_key]
        yaml_cfg.pop('paths')

    def parse_config(self):
        with open(self.cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)

        self.pre_adjust_config(yaml_cfg)

        for name, value in yaml_cfg.items():
            assert hasattr(self, name), f'data class miss attr {name}'
            setattr(self, name, value)

