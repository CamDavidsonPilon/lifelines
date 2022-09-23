# -*- coding: utf-8 -*-
from typing import *
import numpy as np
from lifelines import utils
import pandas as pd


class Printer:
    def __init__(
        self,
        model,
        headers: List[Tuple[str, Any]],
        footers: List[Tuple[str, Any]],
        justify: Callable,
        header_kwargs: Dict,
        decimals: int,
        columns: Optional[List],
    ):
        self.headers = headers
        self.model = model
        self.decimals = decimals
        self.columns = columns
        self.justify = justify
        self.footers = footers

        for tuple_ in header_kwargs.items():
            self.add_to_headers(tuple_)

    def add_to_headers(self, tuple_):
        self.headers.append(tuple_)

    def print_specific_style(self, style):
        if style == "html":
            return self._html_print()
        elif style == "ascii":
            return self._ascii_print()
        elif style == "latex":
            return self._latex_print()
        else:
            raise ValueError("style not available.")

    def print(self, style=None):
        if style is not None:
            self.print_specific_style(style)
        else:
            try:
                from IPython.display import display

                display(self)
            except ImportError:
                self._ascii_print()

    def _latex_print(self):
        print(self.to_latex())

    def to_latex(self):
        summary_df = self.model.summary
        if self.columns is None:
            columns = summary_df.columns
        else:
            columns = summary_df.columns.intersection(self.columns)
        s = summary_df[columns].style
        s = s.format(precision=self.decimals)
        return s.to_latex()

    def _html_print(self):
        print(self.to_html())

    def to_html(self):
        summary_df = self.model.summary

        decimals = self.decimals
        if self.columns is None:
            columns = summary_df.columns
        else:
            columns = summary_df.columns.intersection(self.columns)

        headers = self.headers.copy()
        headers.insert(0, ("model", "lifelines." + self.model._class_name))

        header_df = pd.DataFrame.from_records(headers).set_index(0)

        header_html = header_df.to_html(header=False, notebook=True, index_names=False)

        summary_html = summary_df[columns].to_html(
            col_space=12,
            index_names=False,
            float_format=utils.format_floats(decimals),
            formatters={
                **{c: utils.format_exp_floats(decimals) for c in columns if "exp(" in c},
                **{"p": utils.format_p_value(decimals)},
            },
        )

        if self.footers:
            footer_df = pd.DataFrame.from_records(self.footers).set_index(0)
            footer_html = "<br>" + footer_df.to_html(header=False, notebook=True, index_names=False)
        else:
            footer_html = ""
        return header_html + summary_html + footer_html

    def to_ascii(self):
        df = self.model.summary
        justify = self.justify
        ci = 100 * (1 - self.model.alpha)
        decimals = self.decimals

        repr_string = ""

        repr_string += repr(self.model) + "\n"
        for string, value in self.headers:
            repr_string += "{} = {}".format(justify(string), value) + "\n"

        repr_string += "\n" + "---" + "\n"

        df.columns = utils.map_leading_space(df.columns)

        if self.columns is not None:
            columns = df.columns.intersection(utils.map_leading_space(self.columns))
        else:
            columns = df.columns

        if len(columns) <= 7:
            # only need one row of display
            first_row_set = [
                "coef",
                "exp(coef)",
                "se(coef)",
                "coef lower %d%%" % ci,
                "coef upper %d%%" % ci,
                "exp(coef) lower %d%%" % ci,
                "exp(coef) upper %d%%" % ci,
                "cmp to",
                "z",
                "p",
                "-log2(p)",
            ]
            second_row_set = []

        else:
            first_row_set = [
                "coef",
                "exp(coef)",
                "se(coef)",
                "coef lower %d%%" % ci,
                "coef upper %d%%" % ci,
                "exp(coef) lower %d%%" % ci,
                "exp(coef) upper %d%%" % ci,
            ]
            second_row_set = ["cmp to", "z", "p", "-log2(p)"]

        repr_string += df[columns].to_string(
            float_format=utils.format_floats(decimals),
            formatters={
                **{c: utils.format_exp_floats(decimals) for c in columns if "exp(coef)" in c},
                **{utils.leading_space("p"): utils.format_p_value(decimals)},
            },
            columns=[c for c in utils.map_leading_space(first_row_set) if c in columns],
        )

        if second_row_set:
            repr_string += "\n\n"
            repr_string += df[columns].to_string(
                float_format=utils.format_floats(decimals),
                formatters={
                    **{c: utils.format_exp_floats(decimals) for c in columns if "exp(" in c},
                    **{utils.leading_space("p"): utils.format_p_value(decimals)},
                },
                columns=utils.map_leading_space(second_row_set),
            )

        with np.errstate(invalid="ignore", divide="ignore"):

            repr_string += "\n" + "---" + "\n"
            for string, value in self.footers:
                repr_string += "{} = {}".format(string, value) + "\n"
        return repr_string

    def _ascii_print(self):
        print(self.to_ascii())

    def _repr_latex_(
        self,
    ):
        return self.to_latex()

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self):
        return self.to_ascii()
