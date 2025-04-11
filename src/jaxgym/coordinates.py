from typing import Optional

import jax_dataclasses as jdc
import jax.numpy as jnp
import jax


@jdc.pytree_dataclass
class XYVector:
    x: float
    y: float
    _one: float = 1.

    def modify(self, x=None, y=None) -> "XYVector":
        loc = locals()
        params = {}
        for key in ('x', 'y'):
            ll = loc.get(key, None)
            if ll is None:
                ll = getattr(self, key)
            params[key] = ll
        return XYVector(**params, _one=self._one)

    def __neg__(self) -> 'XYVector':
        return self.__class__(x=-self.x, y=-self.y, _one=self._one)

    def __add__(self, other: 'XYVector') -> 'XYVector':
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            _one=self._one
        )

    def __sub__(self, other: 'XYVector') -> 'XYVector':
        return self.__class__(
            x=self.x - other.x,
            y=self.y - other.y,
            _one=self._one
        )

    def vec(self) -> jnp.ndarray:
        return jnp.array((self.x, self.y, self._one))


@jdc.pytree_dataclass
class XYCoordinateSystem:
    x_vector: XYVector
    y_vector: XYVector
    origin: XYVector

    def transform(self, coords: XYVector) -> XYVector:
        return coords.modify(
            x=self.origin.x * coords._one + coords.x * self.x_vector.x + coords.y * self.y_vector.x,
            y=self.origin.y * coords._one + coords.x * self.x_vector.y + coords.y * self.y_vector.y,
        )

    def matrix(self) -> jnp.ndarray:
        return jnp.array((
            (self.x_vector.x, self.y_vector.x, self.origin.x),
            (self.x_vector.y, self.y_vector.y, self.origin.y),
            (0., 0., 1.),
        ))

    @classmethod
    def from_matrix(cls, mat: jnp.ndarray) -> 'XYCoordinateSystem':
        return cls(
            origin=XYVector(x=mat[0, 2], y=mat[1, 2]),
            x_vector=XYVector(x=mat[0, 0], y=mat[1, 0]),
            y_vector=XYVector(x=mat[0, 1], y=mat[1, 1]),
        )

    def invert(self) -> 'XYCoordinateSystem':
        mat = self.matrix()
        inv = jnp.linalg.inv(mat)
        return self.from_matrix(inv)

    @classmethod
    def identity(cls) -> 'XYCoordinateSystem':
        return cls(
            x_vector=XYVector(x=1., y=0.),
            y_vector=XYVector(x=0, y=1.),
            origin=XYVector(x=0., y=0.),
        )

    def shift(self, shift_vector: XYVector) -> 'XYCoordinateSystem':
        return XYCoordinateSystem(
            x_vector=self.x_vector,
            y_vector=self.y_vector,
            origin=self.origin + shift_vector,
        )

    def rotate(self, radians, center: Optional[XYVector] = None) -> 'XYCoordinateSystem':
        if center is None:
            center = XYVector(x=0, y=0)

        shifted = self.shift(-center)

        mat = jnp.array([
            (jnp.cos(radians), jnp.sin(radians), 0.),
            (-jnp.sin(radians), jnp.cos(radians), 0.),
            (0., 0., 1.)
        ])
        # FIXME validate
        new_mat = shifted.matrix() @ mat

        return self.__class__.from_matrix(new_mat).shift(center)
