from pathlib import Path

import matplotlib.pyplot as plt

from abc import abstractmethod

from matplotlib.patches import Rectangle, FancyBboxPatch, BoxStyle

block_width = 30
block_height = 10
top_arrow_y = 10 + block_height + 5

v_offset = 250


class Element:

    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

    @abstractmethod
    def draw(self, ax: plt.Axes):
        pass

    @property
    @abstractmethod
    def right(self):
        pass

    @property
    @abstractmethod
    def left(self):
        pass

    @property
    @abstractmethod
    def top(self):
        pass

    @property
    @abstractmethod
    def bot(self):
        pass


class Text(Element):
    def __init__(self, x, y, s, **kwargs):
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'center')
        super().__init__(x, y, **kwargs)
        self.s = s

    def draw(self, ax: plt.Axes):
        ax.text(self.x, self.y, self.s, **self.kwargs)

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def bot(self):
        return self.y


class Block(Element):

    def __init__(self, x, y, w, h, colour, rounded=False, **kwargs):
        kwargs.setdefault('ec', 'k')
        super().__init__(x, y, **kwargs)
        self.w, self.h, self.colour = w, h, colour
        self.rounded = rounded

    def draw(self, ax: plt.Axes):
        kw = self.kwargs.copy()

        if self.colour is None:
            kw['fc'] = (1, 1, 1, 0)
        else:
            kw['fc'] = self.colour

        if self.rounded:
            kw['boxstyle'] = BoxStyle("Round", pad=1)
            kw['width'] = self.w - 1
            kw['height'] = self.h - 1
            kw['xy'] = (self.x, self.y)
            p = FancyBboxPatch(**kw)
        else:
            kw['width'] = self.w
            kw['height'] = self.h
            kw['xy'] = (self.x, self.y)
            p = Rectangle(**kw)

        ax.add_patch(p)

    def text(self, s, **kwargs):
        return Text(self.x + 0.5 * self.w, self.y + 0.5 * self.h, s, **kwargs)

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def bot(self):
        return self.y

    @property
    def top(self):
        return self.y + self.h


class Line(Element):
    def __init__(self, x, y, dx, dy, **kwargs):
        super().__init__(x, y, **kwargs)
        self.dx, self.dy = dx, dy

    def draw(self, ax: plt.Axes):
        kwargs = self.kwargs.copy()
        kwargs.setdefault('marker', None)
        ax.plot([self.x, self.x + self.dx], [self.y, self.y + self.dy], **kwargs)

    @property
    def left(self):
        return self.x + (0 if self.dx > 0 else self.dx)

    @property
    def right(self):
        return self.x + (0 if self.dx < 0 else self.dx)

    @property
    def bot(self):
        return self.y + (0 if self.dy > 0 else self.dy)

    @property
    def top(self):
        return self.y + (0 if self.dy < 0 else self.dy)

    def text(self, s, **kwargs):
        if self.dy == 0:
            kwargs.setdefault('va', 'bottom')
            return Text(self.x + 0.5 * self.dx, self.y, s, **kwargs)
        else:
            raise NotImplementedError()


class Arrow(Line):
    style = '<-'

    def __init__(self, x, y, dx, dy, **kwargs):
        super().__init__(x, y, dx, dy, **kwargs)

    def draw(self, ax: plt.Axes):
        ax.annotate(text='', xy=(self.x, self.y), xytext=(self.x + self.dx, self.y + self.dy),
                    arrowprops=dict(arrowstyle=self.style))


class DoubleArrow(Arrow):
    style = '<->'


class PathArrow(Element):
    def __init__(self, xp, yp, **kwargs):
        super().__init__(xp, yp, **kwargs)

    def draw(self, ax: plt.Axes):
        kw = self.kwargs.copy()
        Arrow(self.x[-2], self.y[-2], self.x[-1] - self.x[-2], self.y[-1] - self.y[-2], **kw).draw(ax)
        for k in ['head_width']:
            kw.pop(k, None)
        ax.plot(self.x[:-1], self.y[:-1], **kw)

    @property
    def left(self):
        return min(self.x)

    @property
    def right(self):
        return max(self.x)

    @property
    def top(self):
        return max(self.y)

    @property
    def bot(self):
        return min(self.y)


def draw(lot, out=None):
    f, ax = plt.subplots()
    for lot_element in lot:
        lot_element.draw(ax)
    ax.set_xlim(min(map(lambda x: x.left, lot)) - 1, max(map(lambda x: x.right, lot)) + 1)
    ax.set_ylim(min(map(lambda x: x.bot, lot)) - 1, max(map(lambda x: x.top, lot)) + 1)

    ax.set_aspect('equal')
    # ax.set_xlim(0,1000)
    # ax.set_ylim(0,1000)
    ax.axis("off")
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches='tight')
    plt.show()


def __time_list(cluster_flag, x_offset=0.0, y_offset=0.0):
    """

    Parameters
    ----------
    cluster_flag

    Returns
    -------
    lot: List of Element
    """
    lot = []
    for i in range(4):
        lot.append(Block(block_width * i + x_offset, 10 + y_offset, block_width, block_height,
                         '#1cb2f5' if cluster_flag else '#ff0000'))
    lot.append(Block(block_width * 4 + x_offset, 10 + y_offset, block_width, block_height,
                     '#1cb2f5' if cluster_flag else '#00ff00'))

    return lot


def make_all_fig():
    # Top picture
    lot = __time_list(True)
    line = DoubleArrow(-.25 * block_width, top_arrow_y, 5.5 * block_width, 0)
    lot.append(line)
    lot.append(line.text(r'$t_x+1$ weeks'))

    # Middle Picture
    lot.extend(__time_list(False, x_offset=v_offset))
    line = DoubleArrow(-.25 * block_width + v_offset, top_arrow_y, 4.5 * block_width, 0)
    lot.append(line)
    lot.append(line.text('$X$'))
    line = DoubleArrow(3.75 * block_width + v_offset, top_arrow_y, 1.5 * block_width, 0)
    lot.append(line)
    lot.append(line.text('$y$'))

    # Bottom Picture
    lot.extend(__time_list(False, x_offset=2 * v_offset))
    line = DoubleArrow(-.25 * block_width + 2 * v_offset, top_arrow_y, 4.5 * block_width, 0)
    lot.append(line)
    lot.append(line.text('$X$'))
    line = DoubleArrow(3.75 * block_width + 2 * v_offset, top_arrow_y, 1.5 * block_width, 0)
    lot.append(line)
    lot.append(line.text(r'$\hat{y}$'))

    # Boxes
    for i in range(3):
        lot.append(Block(x=i * v_offset - block_width, y=-3 * block_height,
                         w=7 * block_width, h=12 * block_height,
                         colour=None, rounded=False))
        lot.append(Text(x=i * v_offset - block_width + 3.5 * block_width,
                        y=-2 * block_height,
                        s=f'Section 2.{i + 2}', fontdict=dict(weight='bold'), ha='center'))

    for i in range(2):
        lot.append(Arrow(6 * block_width + i * v_offset, top_arrow_y, v_offset - 7 * block_width, 0))

    bottom_rev_arrow = -6 * block_height

    lot.append(PathArrow([2 * v_offset + 6 * block_width,
                          2 * v_offset + 7 * block_width,
                          2 * v_offset + 7 * block_width,
                          -2 * block_width,
                          -2 * block_width,
                          -block_width
                          ],
                         [top_arrow_y,
                          top_arrow_y,
                          bottom_rev_arrow,
                          bottom_rev_arrow,
                          top_arrow_y,
                          top_arrow_y], color='k'))

    lot.append(Text(v_offset + 5.5 * 0.5 * block_width, bottom_rev_arrow - 20,
                    s=r'$t \leftarrow t+1$', va='center', ha='center'))

    # Time text
    for vo, s in enumerate(['Clustering', 'Training', 'Predicting']):
        if vo < 2:
            for x, st in zip([0, 2, 4, 5], ['$t-t_x-1$', r'$\ldots$', '$t-1$', '$t$']):
                lot.append(Text(x * block_width + vo * v_offset, 0, st, fontdict=dict(size=6)))
        else:
            for x, st in zip([0, 2, 4, 5], ['$t-t_x$', r'$\ldots$', '$t$', '$t+1$']):
                lot.append(Text(x * block_width + vo * v_offset, 0, st, fontdict=dict(size=6)))
        lot.append(Text(x=0 + vo * v_offset + 2.5 * block_width, y=7 * block_height, s=s,
                        ha='center', va='center', fontdict=dict(size=15)))

    draw(lot, out='results/visuals/ccs_framework/all.pdf')
    draw(lot, out='results/visuals/ccs_framework/all.svg')


if __name__ == '__main__':
    make_all_fig()
