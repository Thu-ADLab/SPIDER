import numpy as np

import spider.visualize as vis
import spider.elements as elm
from spider.interface.BaseBenchmark import DummyBenchmark

ego, obs, lmap = DummyBenchmark.get_environment_presets()
xx = np.linspace(0,40,20)
dummy_traj = elm.Trajectory.from_trajectory_array(
    np.array([xx + ego.x(), np.sin(xx/5) + ego.y()]).T, dt=0.2, calc_derivative=True)


vis.figure()
vis.lazy_draw(ego, obs, lmap, dummy_traj)
vis.show()

