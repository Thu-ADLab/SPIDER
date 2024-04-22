import numpy as np


from spider.utils.transform.relative import RelativeTransformer
import spider.visualize as vis

tf = RelativeTransformer()
tf.abs2rel(2, 2, 3.14 / 3, 1, 1, (1, 1, 3.14 / 6), (1, 0))
print(tf)


n = 20
ego_pose = (1, 1, 3.14 / 6)
xs = np.random.rand(n) * 10
ys = np.random.rand(n) * 10
yaws = np.random.rand(n) * 3.14
# colors = np.random.rand(n,3)
xs_rel, ys_rel, yaws_rel,_,_ = tf.abs2rel(xs, ys, yaws, ego_pose=ego_pose)
vis.figure((13,6))
vis.subplot(1,2,1)
vis.draw_obb([ego_pose[0], ego_pose[1], 5,2, ego_pose[2]])
vis.plot(xs, ys, '.b')
vis.subplot(1,2,2)
vis.plot(xs_rel, ys_rel, '.b')

vis.show()
