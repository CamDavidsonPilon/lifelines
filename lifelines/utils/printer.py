# -*- coding: utf-8 -*-
from typing import *
import numpy as np
from lifelines import utils
import pandas as pd


class Printer:
    def __init__(self, model, headers: List[Tuple[str, Any]], justify: Callable, decimals: int, header_kwargs: Dict):
        self.headers = headers
        self.model = model
        self.decimals = decimals
        self.justify = justify

        for tuple_ in header_kwargs.items():
            self.add_to_headers(tuple_)

    def add_to_headers(self, tuple_):
        self.headers.append(tuple_)

    def print_specific_style(self, style):
        if style == "html":
            return self.html_print()
        elif style == "ascii":
            return self.ascii_print()
        elif style == "latex":
            return self.latex_print()
        else:
            raise ValueError("style not available.")

    def print(self, style=None):
        if style is not None:
            self.print_specific_style(style)
        else:
            try:
                from IPython.core.getipython import get_ipython

                ip = get_ipython()
                if ip and ip.has_trait("kernel"):
                    self.html_print_inside_jupyter()
                else:
                    self.ascii_print()
            except ImportError:
                self.ascii_print()

    def latex_print(self):
        print(self.to_latex())

    def to_latex(self):
        decimals = self.decimals
        return self.model.summary.to_latex(float_format="%." + str(decimals) + "f")

    def html_print_inside_jupyter(self):
        from IPython.display import HTML, display

        display(HTML(self.to_html()))

    def html_print(self):
        print(self.to_html())

    def to_html(self):
        decimals = self.decimals
        summary_df = self.model.summary
        columns = summary_df.columns
        headers = self.headers.copy()
        headers.insert(0, ("model", "lifelines." + self.model._class_name))

        header_df = pd.DataFrame.from_records(headers).set_index(0)
        header_html = header_df.to_html(header=False, notebook=True, index_names=False)

        summary_html = summary_df.to_html(
            float_format=utils.format_floats(decimals),
            formatters={
                **{c: utils.format_exp_floats(decimals) for c in columns if "exp(" in c},
                **{"p": utils.format_p_value(decimals)},
            },
        )

        footers = []
        with np.errstate(invalid="ignore", divide="ignore"):

            try:
                if utils.CensoringType.is_right_censoring(self.model) and self.model._KNOWN_MODEL:
                    footers.append(("Concordance", "{:.{prec}f}".format(self.model.score_, prec=decimals)))
            except AttributeError:
                pass

            try:
                sr = self.model.log_likelihood_ratio_test()
                footers.extend(
                    [
                        ("Concordance", "{:.{prec}f}".format(self.model.concordance_index_, prec=decimals)),
                        (
                            "Log-likelihood ratio test",
                            "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                        ),
                        ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-np.log2(sr.p_value), prec=decimals)),
                    ]
                )
            except AttributeError:
                pass

        if footers:
            footer_df = pd.DataFrame.from_records(footers).set_index(0)
            footer_html = footer_df.to_html(header=False, notebook=True, index_names=False)
        else:
            footer_html = ""
        return header_html + summary_html + footer_html

    def ascii_print(self):

        decimals = self.decimals
        df = self.model.summary
        justify = self.justify

        print(self.model)
        for string, value in self.headers:
            print("{} = {}".format(justify(string), value))

        print(end="\n")
        print("---")

        df.columns = utils.map_leading_space(df.columns)
        columns = df.columns

        if len(columns) <= 7:
            # only need one row of display
            first_row_set = [
                "coef",
                "exp(coef)",
                "se(coef)",
                "coef lower 95%",
                "coef upper 95%",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
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
                "coef lower 95%",
                "coef upper 95%",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
            ]
            second_row_set = ["z", "p", "-log2(p)"]

        print(
            df.to_string(
                float_format=utils.format_floats(decimals),
                formatters={
                    **{c: utils.format_exp_floats(decimals) for c in columns if "exp(" in c},
                    **{utils.leading_space("p"): utils.format_p_value(decimals)},
                },
                columns=[c for c in utils.map_leading_space(first_row_set) if c in columns],
            )
        )

        if second_row_set:
            print()
            print(
                df.to_string(
                    float_format=utils.format_floats(decimals),
                    formatters={
                        **{c: utils.format_exp_floats(decimals) for c in columns if "exp(" in c},
                        **{utils.leading_space("p"): utils.format_p_value(decimals)},
                    },
                    columns=utils.map_leading_space(second_row_set),
                )
            )

        with np.errstate(invalid="ignore", divide="ignore"):

            try:
                print("---")
                if utils.CensoringType.is_right_censoring(self.model) and self.model._KNOWN_MODEL:
                    print("Concordance = {:.{prec}f}".format(self.model.concordance_index_, prec=decimals))
            except AttributeError:
                pass

            try:
                sr = self.model.log_likelihood_ratio_test()
                print(
                    "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                        sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value), prec=decimals
                    )
                )
            except AttributeError:
                pass
        print()
