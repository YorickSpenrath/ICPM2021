import time
from pathlib import Path

from functions import file_functions


class NoProgressShower:

    def __init__(self):
        pass

    def update_post(self, new_post, step=None):
        pass

    def update(self):
        pass

    def terminate(self):
        pass

    @property
    def percentage(self):
        return None


class ProgressShower(NoProgressShower):

    def __init__(self, collection, pre=None, post='',
                 show_percentage=True, show_bar=True, show_etr=True
                 , empty='_', fill='█'):
        """
        Simple progress shower. Consists of a
        Parameters
        ----------
        collection: Number or iterable
            Total number of steps. If iterable, its length is used
        pre: Object
            The string to add at the start of the progress bar
        post: Object
            The string to add at the end of the progress bar

        show_percentage: bool
            Show completed percentage
        show_bar: bool
            Show a bar with completed percentage
        show_etr: bool
            Show the expected remaining time.

        empty: str
            The character to indicated the remaining progress. Ignored for show_bar=False
        fill: str
            The character to indicate the progress. Ignored for show_bar=False

        Notes
        -----
        Expected remaining time is computed using the passed execution time, using the average of the passed steps
        """

        super().__init__()

        # If a string that points to a path, make it a path
        if isinstance(collection, str):
            if Path(collection).exists():
                collection = Path(collection)

        # If a path that points to a folder, iterate the files in that folder
        if isinstance(collection, Path) and collection.is_dir():
            list_of_files = file_functions.list_files(collection)
            list_of_dirs = file_functions.list_dirs(collection)

            if len(list_of_files) == 0 and len(list_of_dirs) == 0:
                raise ValueError('Cannot iterate empty folder')
            elif len(list_of_files) == 0 and len(list_of_dirs) > 0:
                collection = list_of_dirs
            elif len(list_of_files) > 0 and len(list_of_dirs) == 0:
                collection = list_of_files
            elif len(list_of_files) > 0 and len(list_of_dirs) > 0:
                raise ValueError('ProgressShower cannot work on folder that contains folders and files')
            else:
                raise NotImplementedError('Impossible branch')
            # If an integer-like value, pick that as range
        try:
            collection = range(int(collection))
        # If not, assume it is a collection
        except (ValueError, TypeError):
            pass

        self.collection = collection
        self.total_steps = len(collection)

        # Setup for iteration
        self.iter = None
        self.current = 0

        assert isinstance(empty, str) and len(empty) == 1, f'invalid value for empty: {empty}'
        assert isinstance(fill, str) and len(fill) == 1, f'invalid value for fill: {fill}'
        self.empty = empty
        self.fill = fill

        self.pre = '' if (pre is None) else str(pre) + ' '

        self.show_percentage = show_percentage
        self.show_bar = show_bar

        self.start_time = time.process_time()
        self.show_eta = show_etr

        self.finished = False

        self.__post = ''
        self.__eta_process_time = None
        self.update_post(post)

    def update_post(self, new_post, step=None):
        self.__post = str(new_post).ljust(len(self.__post))
        if step is not None:
            self.update(step)
        else:
            self.__draw()

    @property
    def post(self):
        return self.__post

    def update(self, step=1):
        """
        Update the progress.

        Terminates if the new value pass self.total_steps.

        Parameters
        ----------
        step: Number
            Number of steps to take
        """
        if self.finished:
            return

        assert step >= 0
        self.current += step

        if self.show_eta and self.current > 0:
            time_that_has_passed = time.process_time() - self.start_time
            self.__eta_process_time = self.start_time + time_that_has_passed / self.current * self.total_steps

        if self.current >= self.total_steps:
            self.terminate()
        else:
            self.__draw()

    def __draw(self):
        """
        Shows the output
        """
        bar = self.fill * self.percentage + self.empty * (100 - self.percentage) if self.show_bar else ''

        if self.show_percentage or self.show_eta:
            percentage = f' ['
            percentage += f"{self.percentage:02}%" if self.show_percentage else ""

            if self.show_percentage and self.show_eta:
                percentage += ' / '
            if self.__eta_process_time is not None:
                rem = self.__eta_process_time - time.process_time()
                percentage += f'{int(rem // 60):02}:{int(rem % 60):02}'
            percentage += ']'
        else:
            percentage = ''

        print(f'\r{self.pre}{bar}{percentage} {self.__post}', end='', flush=True)

    def terminate(self):
        """
        Terminates the Progress Shower instance.

        Adds new line and prevents further drawing

        """
        self.update_post('')
        self.current = self.total_steps
        self.__draw()
        self.finished = True
        print()

    @property
    def percentage(self):
        return int(self.current / self.total_steps * 100)

    def __iter__(self):
        self.iter = iter(self.collection)
        self.first_polled = False
        return self

    def __next__(self):
        if self.first_polled:
            self.update()
        else:
            self.first_polled = True

        foo = next(self.iter)
        if isinstance(foo, Path):
            self.update_post(foo.name)
        if isinstance(foo, tuple):
            self.update_post(foo[0])
        else:
            self.update_post(str(foo))
        return foo
