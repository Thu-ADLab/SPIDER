from .BasicSampler import BasicSampler
from spider.elements.curves import QuarticPolynomial, QuinticPolynomial


# 变量符号含义：x对t的函数
class QuarticPolyminalSampler(BasicSampler):
    def __init__(self, end_t_candidates, end_dx_candidates):
        '''
        end_dx_candidates: x一阶导的终值候选项
        '''
        super(QuarticPolyminalSampler, self).__init__()
        self.end_t_candidates = end_t_candidates
        self.end_dx_candidates = end_dx_candidates

    def sample(self, start_state):
        xs, dxs, ddxs = start_state
        samples = []
        for dxe in self.end_dx_candidates:
            for te in self.end_t_candidates:
                samples.append(QuarticPolynomial(xs, dxs, ddxs, dxe, 0.0, te))

        return samples


class QuinticPolyminalSampler(BasicSampler):
    def __init__(self, end_t_candidates, end_x_candidates):
        """
        end_x_candidates: x的终值候选项
        """
        super(QuinticPolyminalSampler, self).__init__()
        self.end_t_candidates = end_t_candidates
        self.end_x_candidates = end_x_candidates

    def sample(self, start_state):
        # TODO:QZL:未来可以根据werling论文考虑末态s的影响，用于跟车场景/停车线场景
        xs, dxs, ddxs = start_state
        samples = []
        for xe in self.end_x_candidates:
            for te in self.end_t_candidates:
                samples.append(QuinticPolynomial(xs, dxs, ddxs, xe, 0.0, 0.0, te))

        return samples
