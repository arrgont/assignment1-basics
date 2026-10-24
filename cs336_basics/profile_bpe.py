import cProfile
import pstats

from cs336_basics.BPE_training import train_bpe
from tests.conftest import vocab_size

profile = cProfile.Profile()

vocab, merges = profile.runcall(
    train_bpe,
    input_path=(
        "/Users/zyyy/Desktop/assignment1-basics/"
        "data/TinyStoriesV2-GPT4-train.txt"
    ),
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

stats = pstats.Stats(profile).strip_dirs()

print("\n===按累计耗时排序===")
stats.sort_stats("cumulative").print_stats(20)

print("\n===按函数自身耗时排序===")
stats.sort_stats("tottime").print_stats(20)


#output

# ===按累计耗时排序===
#          10869865766 function calls (10869865668 primitive calls) in 2107.532 seconds
#
#    Ordered by: cumulative time
#    List reduced from 178 to 20 due to restriction <20>
#
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1  783.029  783.029 2107.532 2107.532 BPE_training.py:27(train_bpe)
#         1  540.509  540.509  936.294  936.294 count_pretokens.py:51(count_pretoken)
# 2729015243  210.768    0.000  210.768    0.000 count_pretokens.py:80(<genexpr>)
# 2371612015  206.454    0.000  206.454    0.000 {method 'get' of 'dict' objects}
# 1832031151  102.469    0.000  102.469    0.000 {method 'append' of 'list' objects}
# 2415958225   93.716    0.000   93.716    0.000 {built-in method builtins.len}
# 536592168   48.301    0.000   48.301    0.000 {method 'group' of '_regex.Match' objects}
# 536592168   45.080    0.000   45.080    0.000 {method 'encode' of 'str' objects}
# 9745/9744   24.628    0.003   41.442    0.004 {built-in method builtins.max}
#   2717703    1.175    0.000   28.137    0.000 _main.py:350(finditer)
#   2717707    4.259    0.000   19.215    0.000 _main.py:459(_compile)
# 369218707   16.814    0.000   16.814    0.000 BPE_training.py:51(<lambda>)
#   5435573    3.302    0.000    9.769    0.000 enum.py:1607(__and__)
#   2717703    7.753    0.000    7.753    0.000 {method 'finditer' of '_regex.Pattern' objects}
#  16306745    3.028    0.000    4.752    0.000 enum.py:1589(_get_value)
#   2717707    3.737    0.000    3.737    0.000 _main.py:481(complain_unused_args)
#         4    0.000    0.000    3.628    0.907 _main.py:324(split)
#         4    3.622    0.906    3.622    0.906 {method 'split' of '_regex.Pattern' objects}
#         4    3.495    0.874    3.495    0.874 {method 'decode' of 'bytes' objects}
#  29895718    1.882    0.000    1.882    0.000 {built-in method builtins.isinstance}
#
#
#
# ===按函数自身耗时排序===
#          10869865766 function calls (10869865668 primitive calls) in 2107.532 seconds
#
#    Ordered by: internal time
#    List reduced from 178 to 20 due to restriction <20>
#
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1  783.029  783.029 2107.532 2107.532 BPE_training.py:27(train_bpe)
#         1  540.509  540.509  936.294  936.294 count_pretokens.py:51(count_pretoken)
# 2729015243  210.768    0.000  210.768    0.000 count_pretokens.py:80(<genexpr>)
# 2371612015  206.454    0.000  206.454    0.000 {method 'get' of 'dict' objects}
# 1832031151  102.469    0.000  102.469    0.000 {method 'append' of 'list' objects}
# 2415958225   93.716    0.000   93.716    0.000 {built-in method builtins.len}
# 536592168   48.301    0.000   48.301    0.000 {method 'group' of '_regex.Match' objects}
# 536592168   45.080    0.000   45.080    0.000 {method 'encode' of 'str' objects}
# 9745/9744   24.628    0.003   41.442    0.004 {built-in method builtins.max}
# 369218707   16.814    0.000   16.814    0.000 BPE_training.py:51(<lambda>)
#   2717703    7.753    0.000    7.753    0.000 {method 'finditer' of '_regex.Pattern' objects}
#   2717707    4.259    0.000   19.215    0.000 _main.py:459(_compile)
#   2717707    3.737    0.000    3.737    0.000 _main.py:481(complain_unused_args)
#         4    3.622    0.906    3.622    0.906 {method 'split' of '_regex.Pattern' objects}
#         4    3.495    0.874    3.495    0.874 {method 'decode' of 'bytes' objects}
#   5435573    3.302    0.000    9.769    0.000 enum.py:1607(__and__)
#  16306745    3.028    0.000    4.752    0.000 enum.py:1589(_get_value)
#  29895718    1.882    0.000    1.882    0.000 {built-in method builtins.isinstance}
#   2717703    1.175    0.000   28.137    0.000 _main.py:350(finditer)
#   5435583    1.088    0.000    1.715    0.000 enum.py:695(__call__)
