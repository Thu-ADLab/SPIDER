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
def set_background_color(color, fig:plt.Figure=None):
    if fig is None:
        fig = plt.gcf()
    # todo: 还没写完
    pass

def ego_centric_view(ego_x, ego_y, x_range=(-50.,50.), y_range=(-50.,50.), ax:plt.Axes=None):
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(x_range[0] + ego_x, x_range[1]+ego_x)
    ax.set_ylim(y_range[0] + ego_y, y_range[1]+ego_y)
    return ax

def adjust_canvas():
    plt.figure(figsize=(14, 4))
    plt.axis('equal')
    plt.tight_layout()


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
    def __init__(self, auto_snap=False, auto_snap_intervals=1, max_album_size=None,
                 record_video=False, video_path='./output.avi', **video_kwargs): # max_album_size还没用上

        self.album = []

        self.auto_snap = auto_snap # 是否启用auto_snap机制：循环中每次都调用snap，间隔固定次数才会执行snap
        self.auto_snap_intervals = auto_snap_intervals
        self.snap_count = 0

        self.record_video = False
        if record_video:
            import cv2
            self.record_video = True
            # todo: 下面的内容换成可以由video_kwargs设置，设默认值可以先设一个字典，然后update一下
            # video_path =
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 10
            self.video_settings = (video_path, fourcc, fps)
            self.video_writer = None
            # self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (640, 480))

    def _build_video_writer(self, image_example):
        height, width = image_example.shape[:2]
        import cv2
        self.video_writer = cv2.VideoWriter(*self.video_settings, (width, height))
        return self.video_writer

    def clear(self):
        self.album = []
        self.snap_count = 0

    @property
    def album_size(self):
        return len(self.album)

    def snap(self, ax:plt.Axes):
        if self._snap_trigger() or self.record_video:
            img = np.frombuffer(ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
            try: # todo:这边搞不清楚怎么回事，有时候上面work有时候下面work，暂时先这样子写吧
                bbox = ax.figure.bbox.bounds
                img = img.reshape((int(bbox[3]), int(bbox[2]), -1))
            except:
                w, h = ax.figure.canvas.get_width_height()
                img = img.reshape((h, w, 4))

        if self._snap_trigger():
            # # ax.axis('off')
            # img = np.frombuffer(ax.figure.canvas.buffer_rgba(), dtype=np.uint8)
            # bbox = ax.figure.bbox.bounds
            # img = img.reshape((int(bbox[3]), int(bbox[2]), -1))
            self.album.append(img.copy())
            # ax.axis('on')
            # self.album.append(ax)


        if self.record_video:
            if self.video_writer is None:
                self._build_video_writer(img)
            self.video_writer.write(img.copy()[:,:, 2::-1]) # 2是把透明度通道删去，-1是rgb转bgr

        self.snap_count += 1

    def print(self, nrow, ncol, figsize=None, release_video=True):

        if release_video and not (self.video_writer is None):
            self.video_writer.release()

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
        把相册中每张照片保存下来
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
