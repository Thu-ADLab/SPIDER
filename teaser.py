# import matplotlib.pyplot as plt
# import numpy as np
#
# from spider.planner_zoo.LatticePlanner import LatticePlanner
#
# from spider.elements.map import RoutedLocalMap
# from spider.elements.Box import TrackingBoxList, TrackingBox
# from spider.elements.map import Lane
# from spider.elements.vehicle import VehicleState, Transform, Location, Rotation,Vector3D
# import spider.visualize as vis


__all__ = ['demo', 'teaser']

def teaser():
    return demo() # 执行spider.teaser(), 就会执行demo()

def demo():
    from spider.interface.BaseBenchmark import DummyBenchmark
    from spider.planner_zoo import LatticePlanner

    planner = LatticePlanner({
        "steps": 20,
        "dt": 0.2,
        "end_s_candidates": (20, 40, 60),
        "end_l_candidates": (-3.5, 0, 3.5),
    })

    benchmark = DummyBenchmark()
    benchmark.test(planner)



if __name__ == '__main__':
    demo()

