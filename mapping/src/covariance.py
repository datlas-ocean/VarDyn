"""Implicit covariance operators used by the variational control transform."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


class CovarianceOperator(ABC):
    """Matrix-free covariance and square-root interface."""

    @abstractmethod
    def sqrt(self, x):
        """Apply a covariance square root."""

    @abstractmethod
    def sqrt_transpose(self, x):
        """Apply the transpose of the selected covariance square root."""

    @abstractmethod
    def inverse_sqrt(self, x):
        """Apply the inverse of the selected covariance square root."""

    @abstractmethod
    def inverse(self, x):
        """Apply the inverse covariance."""

    @abstractmethod
    def astype(self, dtype):
        """Return the same operator with numerical arrays in ``dtype``."""

    # Historical aliases kept for callers outside the 4DVar driver.
    def sqr(self, x):
        return self.sqrt(x)

    def invsqr(self, x):
        return self.inverse_sqrt(x)

    def inv(self, x):
        return self.inverse(x)


@dataclass(frozen=True)
class DiagonalCovariance(CovarianceOperator):
    """Covariance with a diagonal standard-deviation square root."""

    sigma: object = 1.0

    def __post_init__(self):
        if self.sigma is None:
            object.__setattr__(self, "sigma", 1.0)

    def sqrt(self, x):
        return jnp.asarray(self.sigma) * x

    def sqrt_transpose(self, x):
        return self.sqrt(x)

    def inverse_sqrt(self, x):
        return x / jnp.asarray(self.sigma)

    def inverse(self, x):
        sigma = jnp.asarray(self.sigma)
        return x / (sigma * sigma)

    def astype(self, dtype):
        return DiagonalCovariance(jnp.asarray(self.sigma, dtype=dtype))


@dataclass(frozen=True)
class BlockDiagonalCovariance(CovarianceOperator):
    """Concatenate independent covariance operators without densification."""

    operators: tuple[CovarianceOperator, ...]
    bounds: tuple[int, ...]

    def __post_init__(self):
        if len(self.bounds) != len(self.operators) + 1 or self.bounds[0] != 0:
            raise ValueError("Block covariance bounds are inconsistent.")

    @classmethod
    def from_sizes(cls, operators: Sequence[CovarianceOperator], sizes: Sequence[int]):
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + int(size))
        return cls(tuple(operators), tuple(bounds))

    def _apply(self, method: str, x):
        x = jnp.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.bounds[-1]:
            raise ValueError(
                f"Block covariance expects size {self.bounds[-1]}; got {x.shape}.")
        return jnp.concatenate([
            getattr(operator, method)(x[start:stop])
            for operator, start, stop in zip(
                self.operators, self.bounds[:-1], self.bounds[1:])
        ])

    def sqrt(self, x):
        return self._apply("sqrt", x)

    def sqrt_transpose(self, x):
        return self._apply("sqrt_transpose", x)

    def inverse_sqrt(self, x):
        return self._apply("inverse_sqrt", x)

    def inverse(self, x):
        return self._apply("inverse", x)

    def astype(self, dtype):
        return BlockDiagonalCovariance(
            tuple(operator.astype(dtype) for operator in self.operators),
            self.bounds)
