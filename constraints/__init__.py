'''
qzl: 包含两部分内容：

第一是ConstraintChecker，主要针对采样算法，即给定一条轨迹，需要给出是否可行
第二是ConstriantFormulator ,主要针对优化算法，即列出约束条件.
另外，可以考虑是不是建立一个ConstraintCollection?保存各类约束条件？可以后面再考虑


'''
from spider.constraints.constraint_checker import *
from spider.constraints.constraint_formulator import *
from spider.constraints.ConstraintCollection import ConstraintCollection

