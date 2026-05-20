"""
src/fuzzification.py
Handles the conversion of numeric ratings to Intuitionistic Fuzzy Values (IFVs).
Explicitly models stakeholder uncertainty and hesitation (π) alongside
membership (μ) and non-membership (ν) using Linear, R-Trapezoidal, or Sigmoidal functions.
"""

import numpy as np
import pandas as pd
import math


class Fuzzifier:
    def __init__(self, method='r-trapezoidal', hesitancy_degree=0.1,
                 scale_min=-1.0, scale_max=5.0, trap_limit_min=0.0, trap_limit_max=4.0):
        """
        Initializes the fuzzification function to transform continuous numeric ratings into IFVs.

        Args:
            method: The intuitionistic fuzzification function ('linear', 'r-trapezoidal', 'sigmoidal', 'linear_hesitation').
            hesitancy_degree: The explicitly defined uncertainty/hesitation degree (π).
            scale_min: The absolute minimum of the rating scale (e.g., -1.0 for RALIC).
            scale_max: The absolute maximum of the rating scale (e.g., 5.0 for RALIC).
            trap_limit_min: Corresponds to parameter 'b' in Equations (3, 4). The point where membership starts rising.
            trap_limit_max: Corresponds to parameter 'c' in Equations (3, 4). The point where membership hits the plateau (1-π).
        """
        self.method = method
        self.pi = hesitancy_degree

        # Core evaluation scale bounds
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        # R-trapezoidal transition limits (parameters b and c)
        self.trap_b = float(trap_limit_min)
        self.trap_c = float(trap_limit_max)

    def to_ifv(self, x):
        """Routes the crisp numeric rating to the selected intuitionistic fuzzification function."""
        if pd.isna(x):
            return (np.nan, np.nan, np.nan)

        if self.method == 'linear':
            return self._linear(x)
        elif self.method == 'sigmoidal':
            return self._sigmoidal(x)
        elif self.method == 'r-trapezoidal':
            return self._rtrapezoidal(x)
        elif self.method == 'r-trapezoidal_example':
            return self._rtrapezoidal_example(x)
        elif self.method == 'linear_hesitation':
            return self._linear_hesitation(x)
        else:
            raise ValueError(f"Unknown fuzzification method: {self.method}")

    def _linear(self, x):
        """Applies a linear intuitionistic fuzzification function."""
        if x <= self.scale_min:
            mu = 0.0
            nu = 1.0
        elif self.scale_min < x < self.scale_max:
            mu = (1 - self.pi) * (x - self.scale_min) / (self.scale_max - self.scale_min)
            nu = (1 - self.pi) * (self.scale_max - x) / (self.scale_max - self.scale_min)
        else:
            mu = 1.0
            nu = 0.0

        pi_val = 1.0 - mu - nu
        return (mu, nu, pi_val)

    def _rtrapezoidal(self, x):
        """
        Applies the R-trapezoidal intuitionistic fuzzification function.
        Calculates membership (μ), non-membership (ν), and hesitation (π)
        according to Equations (3), (4) and (5).
        """
        b = self.trap_b
        c = self.trap_c

        if x <= self.scale_min:
            mu = 0.0
            nu = 1.0
        elif x <= b:
            mu = 0.0
            nu = 1.0 - self.pi
        elif b < x < c:
            mu = (1 - self.pi) * (x - b) / (c - b)
            nu = (1 - self.pi) * (c - x) / (c - b)
        elif c <= x < self.scale_max:
            mu = 1.0 - self.pi
            nu = 0.0
        elif x >= self.scale_max:
            mu = 1.0
            nu = 0.0

        pi_val = 1.0 - mu - nu
        return (mu, nu, pi_val)

    def _rtrapezoidal_example(self, x):
        """
        Applies the R-trapezoidal intuitionistic fuzzification function.
        Calculates membership (μ), non-membership (ν), and hesitation (π)
        according to Equations (3), (4) and (5).
        """
        b = self.trap_b
        c = self.trap_c

        if x <= self.scale_min:
            mu = 0.0
            nu = 1.0 - self.pi
        elif x <= b:
            mu = 0.0
            nu = 1.0 - self.pi
        elif b < x < c:
            mu = (1 - self.pi) * (x - b) / (c - b)
            nu = (1 - self.pi) * (c - x) / (c - b)
        elif c <= x < self.scale_max:
            mu = 1.0 - self.pi
            nu = 0.0
        elif x >= self.scale_max:
            mu = 1.0 - self.pi
            nu = 0.0

        pi_val = 1.0 - mu - nu
        return (mu, nu, pi_val)

    def _sigmoidal(self, x):
        """Applies a sigmoidal intuitionistic fuzzification function centered on the evaluation scale."""
        a = 2.0
        c = (self.scale_max + self.scale_min) / 2.0

        mu = (1 - self.pi) / (1 + math.exp(-a * (x - c)))
        nu = (1 - self.pi) - (1 - self.pi) / (1 + math.exp(-a * (x - c)))

        pi_val = 1.0 - mu - nu
        return (mu, nu, pi_val)

    def _linear_hesitation(self, x, alpha_offset=0.05):
        """Applies a linear fuzzification with a variable hesitation degree centered around the midpoint."""
        mu = (1 - 2 * alpha_offset) * (x - self.scale_min) / (self.scale_max - self.scale_min) + alpha_offset

        midpoint = (self.scale_max + self.scale_min) / 2.0
        if x <= midpoint:
            pi_val = 2 * self.pi * (x - self.scale_min) / (self.scale_max - self.scale_min)
        else:
            pi_val = 2 * self.pi * (self.scale_max - x) / (self.scale_max - self.scale_min)

        nu = 1.0 - mu - pi_val
        return (mu, nu, pi_val)


def convert_to_ifv_matrix(df_numeric, method='r-trapezoidal', pi=0.1,
                          scale_min=-1.0, scale_max=5.0, trap_limit_min=0.0, trap_limit_max=4.0):
    """
    Converts a complete numeric ratings matrix into an IFV ratings matrix
    by applying the selected fuzzifier element-wise.
    """
    fuzzifier = Fuzzifier(method=method, hesitancy_degree=pi,
                          scale_min=scale_min, scale_max=scale_max,
                          trap_limit_min=trap_limit_min, trap_limit_max=trap_limit_max)

    # applymap applies the function element-wise
    return df_numeric.map(fuzzifier.to_ifv)


def defuzzify_to_numeric(ifv_val, min_scale=-1.0, max_scale=5.0):
    """
    Converts an IFV representation (μ, ν, π) back to a crisp numeric value.
    Implements Equation (14) for normalization and Equation (15) for de-normalization.

    Args:
        ifv_val (tuple): The (μ, ν, π) tuple to defuzzify.
        min_scale (float): The absolute minimum of the rating scale (parameter a).
        max_scale (float): The absolute maximum of the rating scale (parameter d).

    Returns:
        float: The crisp numeric rating (maintaining 64-bit float precision).
    """
    if not isinstance(ifv_val, tuple) or np.isnan(ifv_val[0]):
        return np.nan

    mu, nu, pi_val = ifv_val

    # Calculate normalized crisp value r in [0, 1]
    r = mu + (mu * (1.0 - mu - nu))

    # De-normalize back to the original rating scale
    numeric_val = r * (max_scale - min_scale) + min_scale

    return numeric_val

def format_ifv(x):
    """Utility function to format an IFV tuple for console display."""
    if isinstance(x, tuple) and not np.isnan(x[0]):
        return f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})"
    return "NaN"
