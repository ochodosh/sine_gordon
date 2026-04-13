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


def heteroclinic(y: jax.Array) -> jax.Array:
    """Heteroclinic (1-ended) solution H whose end is the x-axis.

    H(y) = 4 arctan(exp(y)), a function of y only, satisfying ΔH = sin(H).
    Range: (0, 2π), with H(0) = π, H → 0 as y → −∞, H → 2π as y → +∞.
    """
    return 4.0 * jnp.arctan(jnp.exp(y))


def r(u: jax.Array) -> jax.Array:
    """Inverse of the heteroclinic solution: r = H⁻¹.

    Derived by inverting H(y) = 4 arctan(exp(y)):
      u/4 = arctan(exp(y))  →  exp(y) = tan(u/4)  →  y = log(tan(u/4))

    So r(u) = log(tan(u/4)), valid for u ∈ (0, 2π).

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
