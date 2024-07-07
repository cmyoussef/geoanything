import os

from nukebridge.gizmos import NukeToolbarGenerator as Base_NukeToolbarGenerator


class NukeToolbarGenerator(Base_NukeToolbarGenerator):
    def __init__(self, dev_mode: bool = False):
        super().__init__(dev_mode=dev_mode)
        self.acronym = 'GAI'

    def get_current_file_path(self):
        return os.path.dirname(__file__)
