import hashlib
import os
from pathlib import Path

from functions import file_functions

"""
Creates, prints, and verifies 'hash_dict's,
Which capture the structure of a folder and save hashes of files in the folder

A Hash Dict is a dictionary, created from a folder. The keys of the Hash Dict are the children (folders and files), and
the corresponding values are: 1) if the key is a folder : the Hash Dict of that folder, 2) if the key is a file : the
Hash of that file.
"""


def __compute_hash_file(fn):
    """
    Compute the sha256 hash of a file

    Parameters
    ----------
    fn: String
        The path for which to compute the Hash

    Returns
    -------
        The Hash of the file in the path

    """
    fn = Path(fn)
    assert fn.is_file() and fn.exists(), 'fn is not a file or does not exist'
    with open(fn, 'rb') as rf:
        m = hashlib.sha256()
        m.update(rf.read())
        return m.digest()


def compute_hash_dict(path):
    """
    Compute the HashDict of a directory recursively

    Parameters
    ----------
    path: String
        The filename or directory to compute

    Returns
    -------
        The HashDict of the directory or the Hash of the file

    """
    assert Path(path).exists()
    if Path(path).is_dir():
        ret_dict = {Path(k).name: compute_hash_dict(k) for k in file_functions.list_dirs(path, True)}
        ret_dict.update(
            {Path(k).name: compute_hash_dict(k) for k in file_functions.list_files(path, True)})
        return ret_dict
    else:
        return __compute_hash_file(path)


def print_hash_dict(hash_dict, pre_text=''):
    """
    Print a complete HashDict recursively

    Parameters
    ----------
    hash_dict: HashDict
        The HashDict to print
    pre_text
        The text to put in front of the lines of the print
    """
    for k, v in hash_dict.items():
        print('{}{}'.format(pre_text, k), end='', flush=True)
        if isinstance(v, dict):
            print()
            print_hash_dict(v, pre_text + '\t')
        else:
            print(': {}'.format(v))


def __verify_hash_recursive(path, hash_dict, pre_text):
    """
    Verify a hash recursively

    Parameters
    ----------
    path: String
        The current path to verify
    hash_dict: HashDict
        The current HashDict to verify
    pre_text: String
        The text to put in front of the verification output

    Returns
    -------
    verified: Boolean
        True if the verification succeeded, False otherwise
    """
    state = True
    # verify all values are in the folder
    for k, v in hash_dict.items():
        sub_path = os.path.join(path, str(k))
        if Path(sub_path).exists():
            if isinstance(v, dict):
                print('\t{}{}'.format(pre_text, k))
                state = __verify_hash_recursive(sub_path, v, pre_text + '\t') and state
            else:
                if v == __compute_hash_file(sub_path):
                    print('\t{}{} HASH OK'.format(pre_text, k))
                else:
                    print('!\t{}{} HASH FAILED'.format(pre_text, k))
                    state = False
        else:
            print('!\t{}{} MISSING'.format(pre_text, k))
            state = False

    for a in set(file_functions.list_files(path, False)).union(file_functions.list_dirs(path, False)) \
            .difference(hash_dict.keys()):
        print('!\t{}{}: NOT IN HASH_DICT'.format(pre_text, a))
        state = False

    return state


def verify_hash(path, hash_dict):
    """
    Verify a directory with a hash_dict

    Parameters
    ----------
    path : String
        The directory to verify
    hash_dict: HashDict
        The hash_dict to verify

    Returns
    -------
    verification: Boolean
        True if the verification passes, False if it fails
    """
    state = __verify_hash_recursive(path, hash_dict, '')
    if state:
        print('\nVERIFICATION PASSED')
    else:
        print('\nVERIFICATION FAILED')
    return state


class IntegrityStructure:

    def __init__(self, fd, hash_file):
        """
        Easily groups together a folder with a HashFile object.

        Parameters
        ----------
        fd : String
            Folder to be checked or computed
        hash_file: String
            Location of the HashFile
        """
        assert isinstance(fd, str)
        self.hashed_folder = fd
        assert isinstance(hash_file, str)
        self.hash_file = hash_file

    def save(self):
        """
        Compute and save the HashDict of the folder
        """
        file_functions.save_obj(compute_hash_dict(self.hashed_folder), self.hash_file)

    def check(self):
        """
        Verify the HashDict of the folder
        """
        verify_hash(self.hashed_folder, file_functions.load_obj(self.hash_file))
