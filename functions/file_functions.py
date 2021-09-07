"""
This module contains some commonly used functions that are not easily accessible by pathlib's Path
"""

import pickle
import shutil
import os
from pathlib import Path


def copyfile(src, des, overwrite=False):
    """
    Copy a file from a source to a destination
    Parameters
    ----------
    src: str or Path
        The source path
    des: str or Path
        The destination path
    Raises
    ------
    AssertionError:
        If ``src`` does not exist. If ``des`` does exist.
    """
    assert Path(src).exists(), f'Source does not exist {src}'
    if Path(des).exists():
        if not overwrite:
            raise FileExistsError(f'Destination already exists {des}')
        else:
            delete(Path(des))
    Path(des).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, des)


def copydir(src, des):
    """
    Copy a folder from a source to a destination
    Parameters
    ----------
    src: str or Path
        The source path
    des: str or Path
        The destination path
    Raises
    ------
    AssertionError:
        If ``src`` does not exist. If ``des`` does exist.
    """
    assert Path(src).exists(), f'Source does not exist : {src}'
    assert not Path(des).exists(), f'Destination already exists : {des}'
    Path(des).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, des)


def delete(fn):
    """
    Delete a file or folder. Does nothing if non-existent
    Parameters
    ----------
    fn: str or Path
        The file to delete
    Notes
    -----
    If the path does not exist, nothing is done. If the path is a file, it is removed. If the path is a folder, the
    folder and it contents are removed.
    """
    if not Path(fn).exists():
        return
    if os.path.isdir(fn):
        shutil.rmtree(fn)
    else:
        os.remove(fn)


def list_files(fd, as_full_paths=True):
    """
    List the files in a directory.
    Parameters
    ----------
    fd: str or Path
        The directory to list
    as_full_paths: bool
        Return the list as absolute paths (True) or as filenames (False)
    Returns
    -------
    files: collection of str or collection of Path
        list of Paths if as_full_paths, otherwise list of str with filenames
    """
    assert Path(fd).exists(), f'Folder does not exist : {fd}'
    return [(Path(os.path.join(fd, f)) if as_full_paths else f) for f in os.listdir(fd) if
            os.path.isfile(os.path.join(fd, f))]


def list_file_leafs(fd, as_full_paths=True):
    """
    List the files in a directory and all its subdirectories

    Parameters
    ----------
    fd: str or Path
        The root directory
    as_full_paths: bool
        Return the list of absolute paths (True) or as relative paths (False)

    Returns
    -------
    files: collection of Path or str
        list of Path if as_full_paths, otherwise list of str with relative paths
    """
    if as_full_paths:
        return list_files(fd, True) + sum([list_file_leafs(fd_sub, True) for fd_sub in list_dirs(fd, True)], [])
    else:
        return list_files(fd, False) + sum(
            [[f'{sub_fd.name}/{x}' for x in list_file_leafs(sub_fd, False)] for sub_fd in list_dirs(fd, True)], [])


def list_dirs_recursively(fd):
    """
    Generate a list of folders that are in the tree of which the given `fd` is the root.

    Parameters
    ----------
    fd: Path or str
        Root node of the folder tree

    Returns
    -------
    tree: iterable of Path
        Collection of Paths to the nodes, except the root
    """
    return [Path(x[0]) for x in os.walk(fd)][1:]


def list_dirs(fd, as_full_paths=True):
    """
    List the directories in a directory.
    Parameters
    ----------
    fd: str or Path
        The directory to list
    as_full_paths: bool
        Return the list as absolute paths (True) or as directory names (False)
    Returns
    -------
    files: iterable of str or iterable of Path
        list of Paths if as_full_paths, otherwise list of str with directory names
    """
    assert Path(fd).exists(), f'Folder does not exist : {fd}'
    return [(Path(os.path.join(fd, f)) if as_full_paths else f) for f in os.listdir(fd) if
            not os.path.isfile(os.path.join(fd, f))]


def save_obj(obj, fn):
    """
    Save an object as a pickle
    Parameters
    ----------
    obj: Object
        The object to save
    fn: String
        The location to save the object
    """
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(fn):
    """
    Load an object from a pickle
    Parameters
    ----------
    fn: the location of the object
    Returns
    -------
    pkl: the object saved as pickle in ``fn``
    """
    assert Path(fn).exists()
    with open(fn, 'rb') as f:
        return pickle.load(f)


def rename(src, des):
    """
    Rename a file
    Parameters
    ----------
    src: String
        old file name
    des: String
        new file name
    Raises
    ------
    FileExistsError:
        If the des file name already exists.
    AssertionError
        If the src file does not exist
    """
    assert Path(src).exists(), f'Source file does not exist : {src}'
    if Path(des).exists():
        raise FileExistsError(f'Destination file already exists : {des}')
    os.rename(src, des)


def line_count(fn):
    return sum(1 for _ in open(fn, 'rb'))
