'''
专门针对控制量的约束检查，别的都是针对轨迹的
在hybrid A*这类图搜索的算法里面用得上
'''

from BaseConstraintChecker import BaseConstraintChecker


class ControlConstraintChecker(BaseConstraintChecker):
    def __init__(self, config):
        super(ControlConstraintChecker, self).__init__(config)
        pass


