from os import listdir
from os.path import isfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def discretize(fn_in, fn_out, n_bits):
    arr = np.array(Image.open(fn_in))

    if str(n_bits).startswith('A'):
        arr = (arr > int(n_bits[1:])) * 255
    else:
        arr = np.floor(arr / 256 * n_bits) * 256 / n_bits
    Image.fromarray(arr.astype(np.uint8)).save(fn_out)


def disc_all(img_fns, fd_out, n_bit_vals):
    for n_bits in n_bit_vals:
        for fn in img_fns:
            discretize(fn, fd_out / (fn.name.rsplit('.', 1)[0] + f'_{n_bits}.png'), n_bits)


def create_disc_pictures(fd_in, fd_out):
    fd_in = Path(fd_in)
    fd_out = Path(fd_out)
    fd_out.mkdir(exist_ok=True, parents=True)
    img_fns = [f for f in listdir(fd_in) if (isfile(fd_in / f) and not f.endswith('ini'))]

    sizes = [Image.open(fd_in / fn).size for fn in img_fns]
    names = img_fns

    for n, s in zip(names, sizes):
        print(n, s)

    for n in ['A1', 'A200', 'A20', 2, 4, 8, 16, 32, 256, 50]:
        fdx = fd_out / f'{n}'
        fdx.mkdir(parents=True, exist_ok=True)
        disc_all(img_fns, fdx, [n])
        f, axarr = plt.subplots(nrows=4, ncols=6)
        for (i, ax), fn in zip(enumerate(axarr.flatten()), [f for f in listdir(fdx) if (isfile(fdx / f))]):
            assert isinstance(ax, plt.Axes)
            ax.imshow(np.array(Image.open(fn)), cmap='gray')
            ax.set_axis_off()
            ax.text(10, 200, str(i + 1), color='r', fontsize=20)
        # axarr.flatten()[-1].set_visible(False)
        assert isinstance(f, plt.Figure)
        f.patch.set_color((0, 0, 0))
        f.suptitle(f'n={n}', color='r', fontsize=20)
        f.set_size_inches(w=15, h=12)
        plt.show()
