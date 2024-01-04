import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Sequence
import numpy as np
# from copy import deepcopy
import pickle
import io

# plt.gca().set_facecolor('white')
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     plt.gca().set_zticks([])
#     plt.gca().set_box_aspect([5, 1, 1.5])
#     plt.gca().grid(False)
#     plt.savefig('./images/figure2.pdf', dpi=300, bbox_inches='tight')
#     plt.savefig('./images/figure2.png', dpi=300, bbox_inches='tight')
def get_figure_tight_layout(fig):
    bbox = fig.get_tightbbox(renderer=fig.canvas.get_renderer())

    # 打印边界框的左、右、顶部和底部坐标
    print(f"Left: {bbox.x0}")
    print(f"Right: {bbox.x1}")
    print(f"Top: {bbox.y1}")
    print(f"Bottom: {bbox.y0}")

def no_tick(ax=None):
    ax:plt.Axes = plt.gca() if ax is None else ax
    ax.set_xticks([])
    ax.set_yticks([])
    if isinstance(ax, Axes3D):
        ax.set_zticks([])

def _pkl_deep_copy(data):
    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return pickle.load(buf)

class SnapShot:
    # def __init__(self, max_row=4, max_col=4, max_album_size=None): # max_album_size还没用上
    #     self.max_row = max_row
    #     self.max_col = max_col
    #     self.max_num_one_fig = max_row * max_col
    #     self.album: List[plt.Axes] = []
    def __init__(self, auto_snap=False, auto_snap_intervals=1, max_album_size=None): # max_album_size还没用上

        self.album = [] # : List[plt.Axes]

        self.auto_snap = auto_snap # 是否启用auto_snap机制：循环中每次都调用snap，间隔固定次数才会执行snap
        self.auto_snap_intervals = auto_snap_intervals
        self.snap_count = 0

    def clear(self):
        self.album = []
        self.snap_count = 0

    @property
    def album_size(self):
        return len(self.album)

    def snap(self, ax:plt.Axes):
        if self._snap_trigger():
            ax.axis('off')
            img = np.frombuffer(ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
            bbox = ax.figure.bbox.bounds
            img = img.reshape((int(bbox[3]), int(bbox[2]), -1))
            self.album.append(img.copy())
            ax.axis('on')
            # self.album.append(ax)
        self.snap_count += 1

    def print(self, nrow, ncol, figsize=None):
        max_num_one_fig = nrow * ncol
        album_size = self.album_size

        figure_num = (album_size-1) // max_num_one_fig + 1
        for page in range(figure_num):
            start_idx = page * max_num_one_fig
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
            fig.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0, wspace=0)
            # plt.figure()
            # plt.tight_layout()
            for add_idx, ax in enumerate(axes.flatten()):
                if start_idx+add_idx < album_size:
                    # ax:plt.Axes
                    ax.imshow(self.album[start_idx+add_idx])

                    # ax0 = self.album[start_idx+add_idx]
                    # for artist in ax0.patches: ax.add_patch(_pkl_deep_copy(artist))
                    # for artist in ax0.collections: ax.add_collection(artist)
                    # for artist in ax0.lines: ax.add_line(_pkl_deep_copy(artist))
                # ax.set_frame_on(False)
                ax.axis('off')
                no_tick(ax)
        return


    def save(self):
        '''
        把相册保存下来
        '''
        pass

    def _snap_trigger(self) -> bool:
        # 返回是否触发执行snap
        # 是否启用了auto_snap机制，如果没启用直接snap
        # 循环中每次都调用snap，间隔固定次数才会执行snap
        if self.auto_snap:
            return self.snap_count % self.auto_snap_intervals == 0
        else:
            return True




if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # # from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
    #
    # import math
    #
    # x = range(10)
    # y = range(10, 20)
    #
    # plt.plot(x, y)
    # ax=plt.gca()
    # fig=plt.gcf()
    # img = np.frombuffer(ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
    # ax.figure.canvas.copy_from_bbox(ax.figure.get_tightbbox())
    # img2 = np.frombuffer(ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
    # # plt.savefig("./new_result.png", dpi=120, bbox_inches='tight')
    # plt.show()
