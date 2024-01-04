from spider.visualize.surface3d import *

# 创建一个三维坐标轴
fig = plt.figure()
axes = plt.axes(projection="3d")
top_vertices = np.array([[2, 0], [2, 1],  [3, 1],[3, 0]])
bottom_vertices = np.array([[0, 0], [0, 1],  [1, 1], [1, 0]])

draw_prism(bottom_vertices, 0, top_vertices, 8)
plt.show()
