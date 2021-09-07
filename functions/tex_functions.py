import os
import time

import numpy as np
import subprocess
from pathlib import Path
import pandas as pd
from functions import file_functions
from functions.general_functions import listified

"""
Library for the writing as and production of LaTeX typeset.
"""


# Document with multiple tables ----------------------------------------------------------------------------------------
# noinspection SpellCheckingInspection
class MultipleTablesDocument:

    def __init__(self, destination, centering=True, separate_pages=False):
        self.text = '\n'.join([r'\documentclass{article}', r'\usepackage[center]{titlesec}', r'\usepackage{booktabs}',
                               r'\usepackage[section]{placeins}', r'\usepackage{caption}',
                               r'\usepackage[margin=0.5in]{geometry}', r'\begin{document}', ])
        self.destination = Path(destination)
        self.centering = centering
        self.separate_pages = separate_pages
        self.pre_text = ''

    def add(self, df, caption=None, header=True, index=True, **kwargs):
        # OPTIMIZE : see if we can reduce whitespace from these tables

        assert isinstance(df, pd.DataFrame)

        self.text += self.pre_text
        self.text += df_2_tex(df, caption=caption, label=None, header=header,
                              index=index, centering=self.centering, remove_table_number=True, **kwargs)
        self.text += '\n'
        if self.separate_pages:
            self.pre_text += '\\newpage\n'
        return self

    def produce(self):
        self.text += '\n\n' + r'\end{document}'
        Path(self.destination).parent.mkdir(parents=True, exist_ok=True)
        target_file = Path(self.destination)
        tex_file = target_file.parent / (target_file.name.replace('.pdf', '.tex'))
        with open(tex_file, 'w+') as wf:
            wf.write(self.text)
        tex_2_pdf(tex_file, target_file)
        file_functions.delete(tex_file)


# Conversion to tex ----------------------------------------------------------------------------------------------------

# TODO combine the following two better

def __df_convert(df,
                 add_phantom=False, phantom_length=4, phantom_column_position=0,
                 max_string_length=1000,
                 **kwargs):
    # TODO pandas 1.0.0 also adds label and caption
    assert isinstance(df, pd.DataFrame)

    if add_phantom:
        # TODO add phantom on multiple columns
        phantom_column = f'\\phantom{{{"x" * phantom_length}}}'
        if isinstance(df.columns, pd.MultiIndex):
            phantom_column = (phantom_column,) * df.columns.nlevels
        columns = list(df.columns)[:phantom_column_position] + [phantom_column] + list(df.columns)[
                                                                                  phantom_column_position:]
        df[phantom_column] = ''
        df = df[columns]

    kwargs.setdefault('escape', True)
    with pd.option_context("max_colwidth", max_string_length):
        return df.to_latex(**kwargs)


def df_2_tex(df, fn_out=None, caption=None, label=None, centering=True, floating='h',
             remove_table_number=False, multicolumn=False,
             **kwargs):
    from pylatexenc import latexencode

    s = r'\begin{table'
    if multicolumn:
        s += '*'
    s += '}[' + floating + ']\n'

    if centering:
        s += r'\centering' + '\n'

    s += __df_convert(df, **kwargs)

    if caption is not None:
        s += r'\caption'
        s += '' if not remove_table_number else '*'
        s += '{'
        if kwargs['escape']:
            s += latexencode.unicode_to_latex(caption)
        else:
            s += caption
        s += '}\n'

    if label:
        s += r'\label{' + label + '}\n'

    s += r'\end{table'
    if multicolumn:
        s += '*'
    s += r'}'

    return __ret_or_write(s, fn_out)


# noinspection SpellCheckingInspection
def df_2_standalone_tex(df, fn_out=None, **kwargs):
    # TODO hackish?
    kwargs.pop('caption', None)
    kwargs.pop('label', None)
    kwargs.pop('floating', None)
    s = r'''\documentclass[convert]{standalone}
    \usepackage{booktabs}
    '''

    for p in listified(kwargs.pop('usepackage', []), str):
        s += rf'\usepackage{{{p}}}\n'

    s += r'''\begin{document}'''

    s += __df_convert(df, **kwargs)

    s += r'\end{document}'

    __ret_or_write(s, fn_out)


def __ret_or_write(tex_str, fn_out):
    if fn_out is not None:
        Path(fn_out).parent.mkdir(exist_ok=True, parents=True)
        with open(fn_out, 'w+') as wf:
            wf.write(tex_str)
    else:
        return tex_str


# File conversion ------------------------------------------------------------------------------------------------------

def tex_2_pdf(tex_file, pdf_file):
    # Apparently, you can't just write anywhere. This is now solved by creating a temporary pdf file, and then
    # copying it to the desired location.
    out_ext = 'pdf'
    temp_pdf = Path('temp.' + out_ext)
    ext_less = Path('temp')

    # Execute latex
    x = '(Making PDF)'
    print(x, end='', flush=True)

    process = subprocess.Popen([
        'latex',
        '-output-format=' + out_ext,
        '-quiet',
        '-job-name=' + str(ext_less),
        str(tex_file)])
    process.wait()

    print(f'\r{" " * len(x)}\r', end='', flush=True)

    # cleanup
    for ext in ['aux', 'log']:
        file_functions.delete(ext_less.parent / (ext_less.name + '.' + ext))
    file_functions.copyfile(temp_pdf, pdf_file)
    file_functions.delete(temp_pdf)


def pdf_2_svg(pdf_file, svg_file, page_nr=1):
    assert Path(pdf_file).name.endswith('pdf')
    assert Path(svg_file).name.endswith('svg')
    assert Path(pdf_file).exists()
    Path(svg_file).parent.mkdir(exist_ok=True, parents=True)
    process = subprocess.Popen([
        str(Path(__file__).parent / 'pdf2svg' / 'pdf2svg.exe'),
        str(f'{pdf_file}'),
        str(f'{svg_file}'),
        str(page_nr)])
    process.wait()


# Multiple Conversion --------------------------------------------------------------------------------------------------

def df_2_svg(df, fn_out, **kwargs):
    temp_folder = Path('temp')
    if temp_folder.exists():
        file_functions.delete(temp_folder)
    temp_folder.mkdir(parents=True, exist_ok=False)
    tex_file = temp_folder / 'temp.tex'
    df_2_standalone_tex(df, tex_file, **kwargs)
    pdf_file = temp_folder / 'temp.pdf'
    tex_2_pdf(tex_file, pdf_file)
    pdf_2_svg(pdf_file, fn_out)

    file_functions.delete(temp_folder)
