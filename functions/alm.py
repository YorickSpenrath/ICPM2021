from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from cjo.base.stringconstants import SETTING, VALUE
from functions import dataframe_operations


def _from_xml(fn):
    from xml.etree.ElementTree import Element, parse
    if isinstance(fn, Path):
        e = parse(fn).getroot()
    elif isinstance(fn, Element):
        e = fn
    else:
        raise TypeError(f'Unknown type: {type(fn)}')
    d = dict()
    for child in iter(e):
        if child.tag == 'parameter':
            d.update(child.items())
        else:
            d[child.tag] = _from_xml(child)
    return d


class AbstractSettings(ABC):

    @staticmethod
    def _bool(v):
        if (v == 'False') or (v is False):
            return False
        elif (v == 'True') or (v is True):
            return True
        else:
            raise ValueError(f'Not a bool value: {v}')

    @staticmethod
    def _optional_none(v, t):
        if v == 'None':
            return None
        elif v is None:
            return None
        else:
            return t(v)

    @staticmethod
    def int_or_none(v):
        if v == 'None':
            return None
        elif v is None:
            return None
        else:
            return int(v)

    def convert_to_dict(self, source):
        if source is None:
            return dict()
        if isinstance(source, type(self)):
            sr = source.as_dict()
        elif isinstance(source, Path) or isinstance(source, str):
            source = Path(source)
            if source.is_dir():
                source /= '_settings.csv'
            if source.name.endswith('.xml'):
                return _from_xml(source)
            sr = dataframe_operations.import_sr(source).to_dict()
        elif isinstance(source, pd.Series):
            sr = source.to_dict()
        elif isinstance(source, dict):
            sr = source
        elif source is None:
            sr = dict()
        else:
            # Unknown
            raise TypeError(f'Unknown type: {type(source)}')

        return sr

    def as_dict(self):
        return self.__dict__.copy()

    def __init__(self, source=None):
        d = self.convert_to_dict(source)
        self._assign(d)
        if len(d) > 0:
            raise ValueError(f'Unknown settings : {d.keys()}')

    @abstractmethod
    def _assign(self, d):
        pass

    @property
    @abstractmethod
    def _default_dict(self):
        pass

    def _pop_or_default(self, d, s):
        v = d.pop(s, self._default_dict[s])
        if v == "None":
            v = None
        return v

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.__dict__ == other.__dict__

    def export(self, fn):
        if fn.name.endswith('.xml'):
            self.export_xml(fn)
        elif fn.name.endswith('.csv'):
            self.export_csv(fn)
        else:
            raise NotImplementedError(f'Not implemented for export as {fn.name.split(".")[-1]}')

    def export_csv(self, fn):
        sr = pd.Series(self.__dict__).sort_index()
        sr.index.name = SETTING
        sr.name = VALUE
        dataframe_operations.export_df(sr, fn)

    def export_xml(self, fn):
        from xml.dom import minidom
        from xml.etree import ElementTree

        Path(fn).parent.mkdir(exist_ok=True, parents=True)

        with open(fn, 'w+') as wf:
            wf.write(minidom.parseString(ElementTree.tostring(self.as_element, 'utf-8')).toprettyxml(indent="\t"))

    @property
    def as_element(self):
        from xml.etree.ElementTree import Element
        e = Element(type(self).__name__)
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractSettings):
                sub = Element(k)
                sub.extend(list(v.as_element))
            else:
                sub = Element('parameter')
                if v is None:
                    continue
                else:
                    sub.set(k, str(v))
            e.append(sub)
        return e

    def print(self, pre=''):
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractSettings):
                print(f'{pre}{k}')
                v.print(pre=pre + '\t')
            else:
                print(f'{pre}{k}={v}')

    def __contains__(self, i):
        return i in self.__dict__

    def __getitem__(self, item):
        return self.__dict__[item]


class AbstractLocationsManager(ABC):

    def __init__(self, name, **kwargs):
        """
        Class to take care of locations in an experiment

        Parameters
        ----------
        name: str, Path, LocationsManager
            Name of the experiment, and the resulting save location. Extracts from Path/LM if necessary

        """
        if isinstance(name, str):
            self.name = str(name)
        elif isinstance(name, type(self)):
            self.name = name.name
        elif isinstance(name, Path):
            self.name = name.name
        else:
            raise NotImplementedError(type(name))

        self.fd.mkdir(exist_ok=True, parents=True)

        # Verify settings
        new_settings = self.settings_class(kwargs.copy())
        if not self.fn_settings.exists():
            new_settings.export(self.fn_settings)
        else:
            if len(kwargs) == 0:
                pass
            else:
                assert new_settings == self._settings, f'given parameters do not match saved parameters {name}:{kwargs}'

        # This ensures that added parameters are properly saved
        self._settings.export(self.fn_settings)

    @property
    @abstractmethod
    def settings_class(self):
        pass

    @property
    @abstractmethod
    def fd_base(self):
        pass

    @property
    def fd(self):
        return self.fd_base / self.name

    @property
    def fd_common(self):
        return self.fd_base / 'common'

    @property
    def fn_settings(self):
        return self.fd / 'settings.xml'

    @property
    def _settings(self):
        return self.settings_class(self.fn_settings)

    def __str__(self):
        return f'{self.__class__}[{self.name}]'
