"""Utility functions used."""

import sys
import numpy as np
import odl


class Logger(object):
    """Helper class in order to print output to disc."""
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


class CallbackPrintDiff(odl.solvers.util.callback.SolverCallback):

    """Print data miss-match measured by given functional."""

    def __init__(self, data_func, *args, **kwargs):
        """Initialize a new instance.

        Additional parameters are passed through to the ``show`` method.

        Parameters
        ----------
        file_prefix : string
            Path to where the figure is to be saved
        display_step : positive int, optional
            Number of iterations between plots/saves. Default: 1

        Other Parameters
        ----------------
        kwargs :
            Optional arguments passed on to ``x.show``
        """
        self.args = args
        self.kwargs = kwargs
        self.fig = kwargs.pop('fig', None)
        self.display_step = kwargs.pop('display_step', 1)
        self.iter = 0
        self.data_func = data_func

    def __call__(self, x):
        """Show and save the current iterate."""
        if (self.iter % self.display_step) == 0:
            print('Data missmatch (should be 1):', self.data_func(x))

        self.iter += 1


def Round_To_n(x, n):
    """Helper function to round to the n-th decimal"""
    x = np.float(x)
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n - 1)


class CallbackShowAndSave(odl.solvers.util.callback.SolverCallback):

    """Show and save the iterates."""

    def __init__(self, *args, **kwargs):
        """Initialize a new instance.

        Additional parameters are passed through to the ``show`` method.

        Parameters
        ----------
        file_prefix : string
            Path to where the figure is to be saved
        display_step : positive int, optional
            Number of iterations between plots/saves. Default: 1

        Other Parameters
        ----------------
        kwargs :
            Optional arguments passed on to ``x.show``
        """
        self.file_prefix = kwargs.pop('file_prefix', None)
        self.args = args
        self.kwargs = kwargs
        self.fig = kwargs.pop('fig', None)
        self.display_step = kwargs.pop('display_step', 1)
        self.iter = 0
        self.show_funcs = kwargs.pop('show_funcs', None)

    def __call__(self, x):
        """Show and save the current iterate."""
        if (self.iter % self.display_step) == 0:
            title_string = ''
            if self.show_funcs is not None:
                tmp_vals = [Round_To_n(func(x), 6) for func in self.show_funcs]
                title_string = 'Func values =' + str(tmp_vals)

            if self.file_prefix is not None:
                self.fig = x.show(title=title_string, *self.args, fig=self.fig,
                                  **self.kwargs, saveto=(self.file_prefix +
                                                         '_' + str(self.iter)))
            else:
                self.fig = x.show(title=title_string, *self.args, fig=self.fig,
                                  **self.kwargs)

        self.iter += 1

    def reset(self):
        """Set `iter` to 0 and create a new figure."""
        self.iter = 0
        self.fig = None
