"""
2n-ended solutions to the elliptic sine-Gordon equation Δu = sin(u) on R².

Reference: Liu & Wei, "Classification of Finite Morse Index Solutions to the
Elliptic Sine-Gordon Equation in the Plane" (2021), §2.

The solution family U_n depends on parameters:
  p_j, q_j  with  p_j² + q_j² = 1,  j = 1, …, n   (end directions)
  η_j⁰  ∈ ℝ,                          j = 1, …, n   (phase offsets)

Formula (Eq. 2.15, shifted by π):
  U_n(x, y) = 4 arctan(g_n / f_n)

where f_n, g_n are the even/odd-subset tau functions built from the
interaction coefficients α(j,k) via the Hirota direct method (Eqs. 2.4–2.9).
The shift by π relative to the paper's formula converts the convention
−ΔU = sin(U) to ΔU = sin(U) and places solutions in (0, 2π).

All operations are JAX-compatible and fully differentiable w.r.t. (x, y)
and all parameters (p, q, η⁰).  The subset enumeration (`itertools.combinations`)
runs at JAX trace time (n is always a static Python int), so JIT produces an
unrolled, optimal computation graph for each fixed n.

Numerical stability: f_n and g_n are computed in log-space via logsumexp,
then combined through arctan2 with a shared normalisation factor, avoiding
overflow for large |x|, |y|.
"""

from itertools import combinations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def alpha_matrix(p: jax.Array, q: jax.Array) -> jax.Array:
    """Pairwise interaction coefficients α(j, k) (Eq. 2.4).

    α(j, k) = [(p_j − p_k)² + (q_j − q_k)²] / [(p_j + p_k)² + (q_j + q_k)²]

    Args:
        p: (n,) direction cosines, p_j² + q_j² = 1.
        q: (n,) direction sines,   p_j² + q_j² = 1.

    Returns:
        (n, n) array.  Diagonal entries are 0 (j = k is excluded by assumption).
    """
    dp = p[:, None] - p[None, :]
    dq = q[:, None] - q[None, :]
    sp = p[:, None] + p[None, :]
    sq = q[:, None] + q[None, :]
    return (dp ** 2 + dq ** 2) / (sp ** 2 + sq ** 2)


# ---------------------------------------------------------------------------
# Main solution
# ---------------------------------------------------------------------------

def u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """2n-ended solution to Δu = sin(u).

    Args:
        p:    (n,) direction cosines, must satisfy p[j]² + q[j]² = 1.
        q:    (n,) direction sines,   must satisfy p[j]² + q[j]² = 1.
        eta0: (n,) phase offsets η_j⁰.
        x:    (...) x-coordinates (arbitrary shape).
        y:    (...) y-coordinates (same shape as x).

    Returns:
        U_n of shape (...), values in (0, 2π), satisfying ΔU_n = sin(U_n).
    """
    n = p.shape[0]
    alpha = alpha_matrix(p, q)  # (n, n)

    # Phase variables η_j(x, y) = p_j·x + q_j·y + η_j⁰, shape (n, *x.shape)
    extra = (1,) * x.ndim
    eta = (
        p.reshape((-1,) + extra) * x[None]
        + q.reshape((-1,) + extra) * y[None]
        + eta0.reshape((-1,) + extra)
    )  # (n, ...)

    # For each subset S ⊆ {0,…,n−1}:
    #   log_term(S) = log(a(S)) + Σ_{j∈S} η_j
    # f_n = Σ_{|S| even} a(S) exp(Σ η_j)  →  logsumexp over even log_terms
    # g_n = Σ_{|S| odd}  a(S) exp(Σ η_j)  →  logsumexp over odd  log_terms
    #
    # Empty subset: a(∅) = 1, Σ η = 0  →  log_term = 0 (scalar zero broadcast)

    even_log_terms = [jnp.zeros_like(x)]  # empty subset
    odd_log_terms: list[jax.Array] = []

    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            # log a(subset) = Σ_{j<k in subset} log α(j,k)
            log_a: jax.Array = jnp.zeros(())
            for j, k in combinations(subset, 2):
                log_a = log_a + jnp.log(alpha[j, k])

            eta_sum = sum(eta[j] for j in subset)  # shape (...)
            log_term = log_a + eta_sum

            if size % 2 == 0:
                even_log_terms.append(log_term)
            else:
                odd_log_terms.append(log_term)

    # logsumexp along the new leading axis of stacked terms → shape (...)
    log_f = jax.nn.logsumexp(jnp.stack(even_log_terms, axis=0), axis=0)
    log_g = jax.nn.logsumexp(jnp.stack(odd_log_terms, axis=0), axis=0)

    # arctan2(g, f) = arctan2(exp(log_g − m), exp(log_f − m))  for any m.
    # Choosing m = max(log_g, log_f) keeps the larger argument at 1.0,
    # preventing overflow while preserving the ratio exactly.
    m = jnp.maximum(log_f, log_g)
    return 4.0 * jnp.arctan2(jnp.exp(log_g - m), jnp.exp(log_f - m))


def double_well(t: jax.Array) -> jax.Array:
    """Double-well potential W(t) = 1 - cos(t).

    The sine-Gordon PDE is the Euler-Lagrange equation for this potential:
      W'(t) = sin(t),  so  Δu = sin(u) = W'(u).

    Wells at t = 2πk, k ∈ ℤ.  For u ∈ (0, 2π), the relevant wells are
    t = 0 and t = 2π (identified), with a single barrier at t = π.
    """
    return 1.0 - jnp.cos(t)


def heteroclinic(y: jax.Array) -> jax.Array:
    """Heteroclinic (1-ended) solution H whose end is the x-axis.

    H(y) = 4 arctan(exp(y)), a function of y only, satisfying ΔH = sin(H).
    Range: (0, 2π), with H(0) = π, H → 0 as y → −∞, H → 2π as y → +∞.
    """
    return 4.0 * jnp.arctan(jnp.exp(y))


def heteroclinic_inverse(u: jax.Array) -> jax.Array:
    """Inverse of the heteroclinic solution: r = H⁻¹.

    Derived by inverting H(y) = 4 arctan(exp(y)):
      u/4 = arctan(exp(y))  →  exp(y) = tan(u/4)  →  y = log(tan(u/4))

    So heteroclinic_inverse(u) = log(tan(u/4)), valid for u ∈ (0, 2π).

    Args:
        u: (...) values in (0, 2π).

    Returns:
        r(u) of the same shape, in ℝ.
    """
    return jnp.log(jnp.tan(u / 4.0))


def u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """2n-ended solution parameterised by end angles.

    Uses p_j = cos(θ_j), q_j = sin(θ_j), automatically satisfying the unit
    constraint.  Convenient for unconstrained optimisation over parameters.

    Args:
        theta: (n,) end-direction angles θ_j (radians).
        eta0:  (n,) phase offsets η_j⁰.
        x, y:  spatial coordinates.

    Returns:
        U_n satisfying ΔU_n = sin(U_n).
    """
    return u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


# ---------------------------------------------------------------------------
# Differential helpers
# ---------------------------------------------------------------------------

def _broadcast_xy(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Broadcast spatial coordinates to a common shape."""
    return jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y))


def _evaluate_pointwise_operator(
    operator,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Vectorise a pointwise operator on spatial coordinates over broadcasted x, y."""
    x_b, y_b = _broadcast_xy(x, y)
    coords = jnp.stack((x_b, y_b), axis=-1).reshape((-1, 2))
    values = jax.vmap(operator)(coords)
    return values.reshape(x_b.shape + values.shape[1:])


def _pointwise_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
):
    """Scalar map xy ↦ U_n(x, y) used for pointwise differentiation."""
    return lambda xy: u_n(p, q, eta0, xy[0], xy[1])


def _grad_hessian_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ∇u and D²u on the broadcasted spatial grid."""
    scalar_u = _pointwise_u_n(p, q, eta0)
    grad = _evaluate_pointwise_operator(jax.grad(scalar_u), x, y)
    hessian = _evaluate_pointwise_operator(jax.hessian(scalar_u), x, y)
    return grad, hessian


def grad_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Spatial gradient ∇u of U_n, returned with shape (..., 2)."""
    scalar_u = _pointwise_u_n(p, q, eta0)
    return _evaluate_pointwise_operator(jax.grad(scalar_u), x, y)


def grad_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm |∇u| of U_n."""
    return jnp.linalg.norm(grad_u_n(p, q, eta0, x, y), axis=-1)


def hessian_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Spatial Hessian D²u of U_n, returned with shape (..., 2, 2)."""
    scalar_u = _pointwise_u_n(p, q, eta0)
    return _evaluate_pointwise_operator(jax.hessian(scalar_u), x, y)


def hessian_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm |D²u| of U_n."""
    return jnp.linalg.norm(hessian_u_n(p, q, eta0, x, y), axis=(-2, -1))


def hessian_u_n_grad(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Vector identified with the covector D²u(∇u, ·)."""
    grad, hessian = _grad_hessian_u_n(p, q, eta0, x, y)
    return jnp.einsum("...ij,...j->...i", hessian, grad)


def hessian_u_n_grad_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm |D²u(∇u, ·)| of U_n."""
    return jnp.linalg.norm(hessian_u_n_grad(p, q, eta0, x, y), axis=-1)


def hessian_u_n_grad_grad(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Quadratic form D²u(∇u, ∇u) of U_n."""
    grad, hessian = _grad_hessian_u_n(p, q, eta0, x, y)
    return jnp.einsum("...i,...ij,...j->...", grad, hessian, grad)


def hessian_u_n_det(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Determinant det(D²u) of U_n."""
    return jnp.linalg.det(hessian_u_n(p, q, eta0, x, y))


def third_derivative_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Third spatial derivative D³u of U_n, returned with shape (..., 2, 2, 2)."""
    scalar_u = _pointwise_u_n(p, q, eta0)
    return _evaluate_pointwise_operator(jax.jacfwd(jax.hessian(scalar_u)), x, y)


def third_derivative_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm |D³u| of U_n."""
    third = third_derivative_u_n(p, q, eta0, x, y)
    return jnp.sqrt(jnp.sum(third ** 2, axis=(-3, -2, -1)))


def modica_quantity_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Modica quantity |∇u|² - 2W(u) for U_n.

    This vanishes identically on a 1D heteroclinic profile.
    """
    u = u_n(p, q, eta0, x, y)
    grad = grad_u_n(p, q, eta0, x, y)
    return jnp.einsum("...i,...i->...", grad, grad) - 2.0 * double_well(u)


def modica_gradient_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Gradient of the Modica quantity for U_n, shape (..., 2).

    Using W'(u) = sin(u),
      ∇(|∇u|² - 2W(u)) = 2(D²u ∇u - sin(u) ∇u),
    so this also vanishes on a heteroclinic.
    """
    u = u_n(p, q, eta0, x, y)
    grad, hessian = _grad_hessian_u_n(p, q, eta0, x, y)
    return 2.0 * (
        jnp.einsum("...ij,...j->...i", hessian, grad)
        - jnp.sin(u)[..., None] * grad
    )


def modica_gradient_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm of the Modica gradient for U_n."""
    return jnp.linalg.norm(modica_gradient_u_n(p, q, eta0, x, y), axis=-1)


def one_dimensional_hessian_residual_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Rank-one Hessian residual 2W(u)D²u - W'(u)∇u⊗∇u for U_n.

    For a heteroclinic profile u(x) = H(a · x + b), we have
      D²u = W'(u) a⊗a,   ∇u⊗∇u = 2W(u) a⊗a,
    so this tensor vanishes identically.
    """
    u = u_n(p, q, eta0, x, y)
    grad, hessian = _grad_hessian_u_n(p, q, eta0, x, y)
    return (
        2.0 * double_well(u)[..., None, None] * hessian
        - jnp.sin(u)[..., None, None] * jnp.einsum("...i,...j->...ij", grad, grad)
    )


def one_dimensional_hessian_residual_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm of the rank-one Hessian residual for U_n."""
    return jnp.linalg.norm(
        one_dimensional_hessian_residual_u_n(p, q, eta0, x, y),
        axis=(-2, -1),
    )


def one_dimensional_third_derivative_residual_u_n(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Third-order residual 2W(u)D³u - W''(u)∇u⊗∇u⊗∇u for U_n.

    Differentiating the 1D identity H'' = W'(H) once gives
      H''' = W''(H) H',
    so this tensor vanishes on a heteroclinic profile.
    """
    u = u_n(p, q, eta0, x, y)
    grad = grad_u_n(p, q, eta0, x, y)
    third = third_derivative_u_n(p, q, eta0, x, y)
    return (
        2.0 * double_well(u)[..., None, None, None] * third
        - jnp.cos(u)[..., None, None, None]
        * jnp.einsum("...i,...j,...k->...ijk", grad, grad, grad)
    )


def one_dimensional_third_derivative_residual_u_n_norm(
    p: jax.Array,
    q: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm of the third-order heteroclinic residual for U_n."""
    residual = one_dimensional_third_derivative_residual_u_n(p, q, eta0, x, y)
    return jnp.sqrt(jnp.sum(residual ** 2, axis=(-3, -2, -1)))


def grad_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Spatial gradient ∇u for the angle-parameterised solution."""
    return grad_u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def grad_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm |∇u| for the angle-parameterised solution."""
    return grad_u_n_norm(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Spatial Hessian D²u for the angle-parameterised solution."""
    return hessian_u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm |D²u| for the angle-parameterised solution."""
    return hessian_u_n_norm(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_grad_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Vector identified with D²u(∇u, ·) for the angle-parameterised solution."""
    return hessian_u_n_grad(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_grad_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm |D²u(∇u, ·)| for the angle-parameterised solution."""
    return hessian_u_n_grad_norm(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_grad_grad_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Quadratic form D²u(∇u, ∇u) for the angle-parameterised solution."""
    return hessian_u_n_grad_grad(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def hessian_u_n_det_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Determinant det(D²u) for the angle-parameterised solution."""
    return hessian_u_n_det(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def third_derivative_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Third spatial derivative D³u for the angle-parameterised solution."""
    return third_derivative_u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def third_derivative_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm |D³u| for the angle-parameterised solution."""
    return third_derivative_u_n_norm(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def modica_quantity_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Modica quantity |∇u|² - 2W(u) for the angle-parameterised solution."""
    return modica_quantity_u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def modica_gradient_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Gradient of the Modica quantity for the angle-parameterised solution."""
    return modica_gradient_u_n(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def modica_gradient_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Euclidean norm of the Modica gradient for the angle-parameterised solution."""
    return modica_gradient_u_n_norm(jnp.cos(theta), jnp.sin(theta), eta0, x, y)


def one_dimensional_hessian_residual_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Rank-one Hessian residual for the angle-parameterised solution."""
    return one_dimensional_hessian_residual_u_n(
        jnp.cos(theta), jnp.sin(theta), eta0, x, y
    )


def one_dimensional_hessian_residual_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm of the rank-one Hessian residual for the angle-parameterised solution."""
    return one_dimensional_hessian_residual_u_n_norm(
        jnp.cos(theta), jnp.sin(theta), eta0, x, y
    )


def one_dimensional_third_derivative_residual_u_n_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Third-order heteroclinic residual for the angle-parameterised solution."""
    return one_dimensional_third_derivative_residual_u_n(
        jnp.cos(theta), jnp.sin(theta), eta0, x, y
    )


def one_dimensional_third_derivative_residual_u_n_norm_from_angles(
    theta: jax.Array,
    eta0: jax.Array,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Frobenius norm of the third-order residual for the angle-parameterised solution."""
    return one_dimensional_third_derivative_residual_u_n_norm(
        jnp.cos(theta), jnp.sin(theta), eta0, x, y
    )
