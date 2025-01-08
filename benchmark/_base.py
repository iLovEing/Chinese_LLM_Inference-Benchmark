from dataclasses import dataclass


@dataclass
class ChoiceBenchmark:
    benchmark: str = None
    name_EN: str = None
    name_CH: str = None

    choices: list[str] = ()
    test_qst: list[str] = ()
    test_ans: list[str] = ()
    dev_qst: list[str] = ()
    dev_ans: list[str] = ()

    def __init__(self,
                 benchmark: str = None,
                 name_EN: str = None,
                 name_CH: str = None,
                 choices: list[str] = (),
                 test_qst: list[str] = (),
                 test_ans: list[str] = (),
                 dev_qst: list[str] = (),
                 dev_ans: list[str] = (),
                 ):
        self.benchmark = benchmark
        self.name_EN = name_EN
        self.name_CH = name_CH
        self.choices = choices
        self.test_qst = test_qst
        self.test_ans = test_ans
        self.dev_qst = dev_qst
        self.dev_ans = dev_ans

        assert len(self.choices) > 1, f'invalid choice: {self.choices}'
        assert 0 < len(self.test_qst) == len(self.test_ans), f'invalid test case len: {len(self.test_qst)}, {len(self.test_ans)}'
        assert len(self.dev_ans) == len(self.dev_qst), f'invalid dev case len: {len(self.dev_qst)}, {len(self.dev_ans)}'

    def __repr__(self):
        result = ''
        result += f'{self.__class__.__name__}\n'
        for k, v in self.__dict__.items():
            result += f'{k}: {v}\n'

        return result.strip()

