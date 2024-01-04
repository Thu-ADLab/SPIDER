from spider.visualize.surface import *

from spider.elements import TrackingBox, TrackingBoxList
tb = TrackingBox(obb=[1,6,5,2,3.14/6])
tb_list = TrackingBoxList([tb])
draw_polygon(tb.vertices, fill=True,lw=1.5, alpha=0.2)#,color='red',

draw_circle([2, 5], 1,mark_center=True, fill=True, alpha=0.6)
draw_circle([3, 7], 1, mark_center=True, color='blue')
plt.show()

