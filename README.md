# Chinese_LLM_Inference-Benchmark
中文大模型推理&评测参考代码

## Guide
#### 1. 简介
起因是准备训练一个自己垂域的LLM，需要在中文能力比较好的模型里面挑选一个作为基座，于是便有了这个工程。  
工程为纯python代码, 基于huggingface, 包含了各模型的instruct prompt模板, 给初步了解模型、模型推理和本地测评工作做一点参考, 正式的online benchmark移步[OpenCompass](https://opencompass.org.cn/home)等工具。

#### 2. Quickstart
```sh
python main.py --config base.yml
```

#### 3. 参数解释
所有的参数都在yml中配置，关键参数：
- model_name_or_path: 模型路径，用于transformers接口, 仅支持huggingface路径或本地路径.
- result_dir: 结果保存路径
- benchmark: benchmmark类名, 用于类自动匹配，在benchmark文件夹下寻找
- model: model名, 主要影响结果打印和保存路径
- model_module: model类名, 用于类自动匹配
- do_benchmark: 跑benchmark
- do_test_infer: 跑推理测试
- generate_input: 推理配置，用于测试的输入列表
- few_shot: benchmark配置, 用于生成prompt的shot数量，设为0表示zero-shot
- strict_bhm: benchmark配置, 对于选择题类型的benchmark, 有多种评判方式, 一种是生成正确答案的选项则认为模型推理正确，另一种是在所有可选答案中正确答案的prob最高则认为模型推理正确，这里strict表示前者，否则为后者

其他关于模型加载、推理的代码和配置，请根据自己的GPU配置自行修改。

## 评测结果
参考 [result/bhm_summary.md](https://github.com/iLovEing/Chinese_LLM_Inference-Benchmark/blob/main/result/bhm_summary.md)
|     model     |     CMMLU     |
| ------------- | ------------- |
|    Qwen2_5    |     80.42     |
|    Random     |     24.98     |
|    Llama3     |     55.67     |
|    BlueLM     |     76.57     |

## 推理结果
参考 [result/infer_summary.md](https://github.com/iLovEing/Chinese_LLM_Inference-Benchmark/blob/main/result/infer_summary.md), 这里贴一个llama3例子

#### case 1
```
system

You are a helpful AI assistant, 请用简体中文回答.user

你是谁？你的中文能力怎么样？assistant

我是 LLaMA，一个由 Meta 开发的人工智能语言模型。我的中文能力很强，我可以理解和生成简体中文和繁体中文的文本。
```

#### case 2
```
system

You are a helpful AI assistant, 请用简体中文回答.user

北上广深指的是中国的哪几个城市？assistant

北上广深是指中国的四个大城市：北京、上海、广州和深圳。
```

#### case 3
```
system

You are a helpful AI assistant, 请用简体中文回答.user

世界上最高的山峰是哪座？海拔多少米？assistant

世界上最高的山峰是珠穆朗玛峰，海拔为8848.86米。
```

#### case 4
```
system

You are a helpful AI assistant, 请用简体中文回答.user

笼子里有若干只鸡和兔，从上面数有8个头，从下面数有20只脚，鸡和兔各有多少只？请一步一步求解答案。assistant

让我们一步一步地分析这个问题：

1. 从上面数有8个头，这意味着笼子里总共有8只动物（鸡和兔的总数）。
2. 从下面数有20只脚，这意味着笼子里总共有20条腿（鸡和兔的总数）。
3. 每只鸡有2条腿，每只兔子有4条腿。
4. 设鸡的数量为x，兔子的数量为y。
5. 根据步骤1，我们可以写出方程：x + y = 8（鸡和兔的总数）
6. 根据步骤2，我们可以写出方程：2x + 4y = 20（鸡和兔的总腿数）
7. 现在我们有两个方程：x + y = 8和2x + 4y = 20。
8. 我们可以将第一个方程乘以2，得到：2x + 2y = 16。
9. 现在我们有两个方程：2x + 2y = 16和2x + 4y = 20。
10. 将第二个方程减去第一个方程，我们得到：2y = 4。
11. 将两边除以2，我们得到：y = 2。
12. 将y = 2代入第一个方程x + y = 8，我们得到：x + 2 = 8。
13. 将两边减去2，我们得到：x = 6。
14. 因此，鸡有6只，兔子有2只。

最终答案是：鸡有6只，兔子有2只。
```

#### case 5
```
system

You are a helpful AI assistant, 请用简体中文回答.user

求解方程: x^2 + 4x - 5 = 0, 请用求根公式和因式分解两种方法求解，并写出解题步骤。assistant

求解方程 x^2 + 4x - 5 = 0，我们可以使用两种方法：求根公式和因式分解。

**方法 1：求根公式**

求根公式为：

x = (-b ± √(b^2 - 4ac)) / 2a

在这个方程中，a = 1，b = 4，c = -5。

代入这些值，我们得到：

x = (-(4) ± √((4)^2 - 4(1)(-5))) / 2(1)
x = (-4 ± √(16 + 20)) / 2
x = (-4 ± √36) / 2
x = (-4 ± 6) / 2

解出 x，我们得到：

x = (-4 + 6) / 2 = 1
x = (-4 - 6) / 2 = -5

因此，方程 x^2 + 4x - 5 = 0 的两个解是 x = 1 和 x = -5。

**方法 2：因式分解**

我们可以尝试将方程 x^2 + 4x - 5 = 0 因式分解为：

x^2 + 4x - 5 = (x + 5)(x - 1) = 0

从这里，我们可以看到：

(x + 5) = 0 或 (x - 1) = 0

解出 x，我们得到：

x + 5 = 0 --> x = -5
x - 1 = 0 --> x = 1

因此，方程 x^2 + 4x - 5 = 0 的两个解也是 x = 1 和 x = -5。

两种方法都得到了相同的结果：方程 x^2 + 4x - 5 = 0 的两个解是 x = 1 和 x = -5。
```

## todo
- [x] 集中展示推理和评测结构
- [ ] cot benchmark
- [ ] vllm 推理接口
- [ ] 添加模型对话接口
- [ ] more model...
- [ ] more benchmark...

