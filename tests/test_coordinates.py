from jaxgym.coordinates import XYVector, XYCoordinateSystem
from numpy.testing import assert_allclose
import jax.numpy as jnp


def test_add():
    v1 = XYVector(7., 13.)
    v2 = XYVector(2., 0.1)

    su = v1 + v2

    assert_allclose(su.x, v1.x + v2.x)
    assert_allclose(su.y, v1.y + v2.y)
    assert_allclose(su._one, 1.)


def test_subtract():
    v1 = XYVector(7., 13.)
    v2 = XYVector(2., 0.1)

    su = v1 - v2

    assert_allclose(su.x, v1.x - v2.x)
    assert_allclose(su.y, v1.y - v2.y)
    assert_allclose(su._one, 1.)


def test_minus():
    v1 = XYVector(7., 13.)

    res = -v1
    ref = XYVector(0, 0) - v1

    assert_allclose(res.x, ref.x)
    assert_allclose(res.y, ref.y)
    assert_allclose(res._one, 1.)


def test_transform():
    coords = XYCoordinateSystem(
        origin=XYVector(7., 13.),
        x_vector=XYVector(2., 0.1),
        y_vector=XYVector(0.5, 1.),
    )

    vec_1 = XYVector(0, 0)
    assert_allclose(
        coords.transform(vec_1).vec(),
        coords.origin.vec(),
    )

    vec_2 = XYVector(1, 0)
    assert_allclose(
        coords.transform(vec_2).vec()[:2],
        coords.origin.vec()[:2] + coords.x_vector.vec()[:2],
    )
    assert_allclose(
        coords.transform(vec_2).vec()[2],
        1.
    )

    vec_3 = XYVector(0, 1)
    assert_allclose(
        coords.transform(vec_3).vec()[:2],
        coords.origin.vec()[:2] + coords.y_vector.vec()[:2],
    )
    assert_allclose(
        coords.transform(vec_3).vec()[2],
        1.
    )


def test_matrix():
    coords = XYCoordinateSystem(
        origin=XYVector(7., 13.),
        x_vector=XYVector(2., 0.1),
        y_vector=XYVector(0.5, 1.),
    )
    mat = jnp.array((
        (2., 0.5, 7.),
        (0.1, 1., 13.),
        (0., 0., 1.),
    ))

    assert_allclose(coords.matrix(), mat)

    test = XYVector(x=17., y=19.)
    transformed_ref = coords.transform(test)
    transformed_res = XYCoordinateSystem.from_matrix(mat).transform(test)

    assert_allclose(transformed_ref.vec(), transformed_res.vec())


def test_inversion():
    coords = XYCoordinateSystem(
        origin=XYVector(7., 13.),
        x_vector=XYVector(2., 0.1),
        y_vector=XYVector(0.5, 1.),
    )
    inverted = coords.invert()
    assert_allclose(inverted.invert().matrix(), coords.matrix(), rtol=1e-6, atol=1e-6)
    test = XYVector(x=17., y=19.)
    transformed = coords.transform(test)
    back = inverted.transform(transformed)

    assert_allclose(test.vec(), back.vec(), rtol=1e-6, atol=1e-6)


def test_identity():
    ident = XYCoordinateSystem.identity()
    eye = jnp.eye(3)
    assert_allclose(ident.matrix(), eye)


def test_shift():
    raise NotImplementedError('TODO')


def test_rotate():
    raise NotImplementedError('TODO')
