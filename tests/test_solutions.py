"""
Tests for the k-ended sine-Gordon solutions (Δu = sin(u), u ∈ (0, 2π)).

Each test checks the Hirota formula against a known closed-form arctan
expression, verifies PDE satisfaction, differentiability, and JIT.
"""

import jax
import jax.numpy as jnp
import pytest

from sine_gordon import heteroclinic, r, u_n, u_n_from_angles

jax.config.update("jax_enable_x64", True)

# Spatial grid used throughout
_xs = jnp.linspace(-4.0, 4.0, 30)
_X, _Y = jnp.meshgrid(_xs, _xs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laplacian_at(f_scalar, x_val: float, y_val: float) -> float:
    """Laplacian of f(x,y) at a point via JAX hessian (sum of diagonal)."""
    xy = jnp.array([x_val, y_val])
    H = jax.hessian(lambda xy_: f_scalar(xy_[0], xy_[1]))(xy)
    return H[0, 0] + H[1, 1]


# ---------------------------------------------------------------------------
# Known closed-form tests (arctan comparisons)
# ---------------------------------------------------------------------------

class TestClosedForms:
    """Check that the Hirota formula reproduces known arctan expressions."""

    def test_n1_vs_heteroclinic(self):
        """n=1: U_1 = 4 arctan(exp(p·x + q·y + η⁰)) − π  (1D heteroclinic).

        This is the unique (up to translation/rotation) 2-ended solution.
        """
        p = jnp.array([3.0 / 5.0])
        q = jnp.array([4.0 / 5.0])
        eta0 = jnp.array([0.7])

        got = u_n(p, q, eta0, _X, _Y)
        expected = 4.0 * jnp.arctan(jnp.exp(p[0] * _X + q[0] * _Y + eta0[0]))

        assert jnp.allclose(got, expected, atol=1e-6), (
            f"max |err| = {jnp.max(jnp.abs(got - expected)):.2e}"
        )

    def test_n2_saddle_vs_eq216(self):
        """n=2 saddle: 4 arctan(cosh(y/√2) / cosh(x/√2)) − π  (Eq. 2.16).

        Parameters: p1=p2=q1=1/√2, q2=−1/√2, η⁰=0.
        """
        s = 1.0 / jnp.sqrt(2.0)
        p = jnp.array([s, s])
        q = jnp.array([s, -s])
        eta0 = jnp.zeros(2)

        got = u_n(p, q, eta0, _X, _Y)
        expected = 4.0 * jnp.arctan(jnp.cosh(_Y * s) / jnp.cosh(_X * s))

        assert jnp.allclose(got, expected, atol=1e-6), (
            f"max |err| = {jnp.max(jnp.abs(got - expected)):.2e}"
        )

    @pytest.mark.parametrize("angle", [jnp.pi / 6, jnp.pi / 4, jnp.pi / 3])
    def test_n2_phi_pq_vs_closed_form(self, angle):
        """n=2 general 4-end family φ_{p,q}.

        Closed form (paper §2, after sign correction):
          φ_{p,q} = 4 arctan(p·cosh(q·y) / (q·cosh(p·x))) − π

        Hirota parameters: p1=p2=p, q1=−q2=q, η1⁰=η2⁰=ln(p/q).
        Note: the paper states η⁰=ln(q/p) but this is a sign typo; ln(p/q)
        is the correct value — verified below by exact pointwise agreement.
        """
        p_val = float(jnp.cos(angle))
        q_val = float(jnp.sin(angle))

        p = jnp.array([p_val, p_val])
        q = jnp.array([q_val, -q_val])
        eta0 = jnp.full(2, jnp.log(p_val / q_val))

        got = u_n(p, q, eta0, _X, _Y)
        expected = 4.0 * jnp.arctan(p_val * jnp.cosh(q_val * _Y) / (q_val * jnp.cosh(p_val * _X)))

        assert jnp.allclose(got, expected, atol=1e-6), (
            f"angle={float(angle):.3f}: max |err| = {jnp.max(jnp.abs(got - expected)):.2e}"
        )


# ---------------------------------------------------------------------------
# PDE residual tests
# ---------------------------------------------------------------------------

class TestPDE:
    """Verify ΔU_n = sin(U_n) numerically via JAX hessian."""

    @pytest.mark.parametrize("n_ends,params", [
        (
            1,
            dict(
                p=jnp.array([3.0 / 5.0]),
                q=jnp.array([4.0 / 5.0]),
                eta0=jnp.array([0.0]),
            ),
        ),
        (
            2,
            dict(
                p=jnp.array([1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)]),
                q=jnp.array([1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)]),
                eta0=jnp.zeros(2),
            ),
        ),
        (
            3,
            dict(
                p=jnp.array([jnp.cos(jnp.pi / 6), jnp.cos(jnp.pi / 3), jnp.cos(jnp.pi / 2)]),
                q=jnp.array([jnp.sin(jnp.pi / 6), jnp.sin(jnp.pi / 3), jnp.sin(jnp.pi / 2)]),
                eta0=jnp.array([0.0, 0.5, -0.5]),
            ),
        ),
    ])
    def test_pde_residual(self, n_ends, params):
        """Max |ΔU_n − sin(U_n)| < 1e-4 on a coarse grid."""
        p, q, eta0 = params["p"], params["q"], params["eta0"]

        # Evaluate on a small set of interior points (hessian is per-point)
        check_pts = [(-2.0, -1.0), (0.0, 0.0), (1.5, 2.0), (-1.0, 3.0)]
        residuals = []
        for x_val, y_val in check_pts:
            f = lambda x_, y_: u_n(p, q, eta0, x_, y_)
            lap = _laplacian_at(f, x_val, y_val)
            u_val = float(u_n(p, q, eta0, jnp.array(x_val), jnp.array(y_val)))
            residuals.append(abs(float(lap) - jnp.sin(u_val)))

        max_res = max(residuals)
        assert max_res < 1e-4, f"n={n_ends}: max PDE residual = {max_res:.2e}"


# ---------------------------------------------------------------------------
# Differentiability tests
# ---------------------------------------------------------------------------

class TestDifferentiability:
    """jax.grad works w.r.t. all inputs; results are finite."""

    def setup_method(self):
        s = 1.0 / jnp.sqrt(2.0)
        self.p = jnp.array([s, s])
        self.q = jnp.array([s, -s])
        self.eta0 = jnp.zeros(2)
        self.x0 = jnp.array(1.0)
        self.y0 = jnp.array(0.5)

    def _scalar_u(self, p, q, eta0, x, y):
        return u_n(p, q, eta0, x, y)

    def test_grad_wrt_x(self):
        grad = jax.grad(self._scalar_u, argnums=3)(
            self.p, self.q, self.eta0, self.x0, self.y0
        )
        assert jnp.isfinite(grad), f"grad_x not finite: {grad}"

    def test_grad_wrt_y(self):
        grad = jax.grad(self._scalar_u, argnums=4)(
            self.p, self.q, self.eta0, self.x0, self.y0
        )
        assert jnp.isfinite(grad), f"grad_y not finite: {grad}"

    def test_grad_wrt_p(self):
        grad = jax.grad(lambda p_: self._scalar_u(p_, self.q, self.eta0, self.x0, self.y0))(
            self.p
        )
        assert jnp.all(jnp.isfinite(grad)), f"grad_p not finite: {grad}"

    def test_grad_wrt_q(self):
        grad = jax.grad(lambda q_: self._scalar_u(self.p, q_, self.eta0, self.x0, self.y0))(
            self.q
        )
        assert jnp.all(jnp.isfinite(grad)), f"grad_q not finite: {grad}"

    def test_grad_wrt_eta0(self):
        grad = jax.grad(
            lambda e: self._scalar_u(self.p, self.q, e, self.x0, self.y0)
        )(self.eta0)
        assert jnp.all(jnp.isfinite(grad)), f"grad_eta0 not finite: {grad}"

    def test_angle_parameterisation_grad(self):
        theta = jnp.array([jnp.pi / 4, 3 * jnp.pi / 4])
        grad = jax.grad(
            lambda th: u_n_from_angles(th, self.eta0, self.x0, self.y0)
        )(theta)
        assert jnp.all(jnp.isfinite(grad)), f"grad_theta not finite: {grad}"


# ---------------------------------------------------------------------------
# JIT test
# ---------------------------------------------------------------------------

class TestJIT:
    def test_jit_compiles_and_runs(self):
        s = 1.0 / jnp.sqrt(2.0)
        p = jnp.array([s, s])
        q = jnp.array([s, -s])
        eta0 = jnp.zeros(2)

        jit_u = jax.jit(u_n)
        result = jit_u(p, q, eta0, _X, _Y)

        assert result.shape == _X.shape
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0) and jnp.all(result < 2 * jnp.pi)


# ---------------------------------------------------------------------------
# Heteroclinic and its inverse r
# ---------------------------------------------------------------------------

class TestHeteroclinicAndR:
    """Tests for H(y) = 4 arctan(exp(y)) and r = H⁻¹."""

    _ys = jnp.linspace(-5.0, 5.0, 200)

    def test_heteroclinic_range(self):
        """H maps ℝ into (0, 2π)."""
        H = heteroclinic(self._ys)
        assert jnp.all(H > 0) and jnp.all(H < 2 * jnp.pi)

    def test_heteroclinic_midpoint(self):
        """H(0) = π."""
        assert jnp.allclose(heteroclinic(jnp.array(0.0)), jnp.pi, atol=1e-7)

    def test_heteroclinic_satisfies_pde(self):
        """H satisfies ΔH = sin(H) (1D: H''(y) = sin(H(y)))."""
        check_ys = [-3.0, -1.0, 0.0, 1.0, 3.0]
        for y_val in check_ys:
            d2H = float(jax.grad(jax.grad(lambda y_: heteroclinic(y_)))(jnp.array(y_val)))
            H_val = float(heteroclinic(jnp.array(y_val)))
            assert abs(d2H - jnp.sin(H_val)) < 1e-5, (
                f"y={y_val}: H''={d2H:.6f}, sin(H)={float(jnp.sin(H_val)):.6f}"
            )

    def test_r_is_inverse_of_H(self):
        """r(H(y)) = y for y ∈ [−5, 5]."""
        roundtrip = r(heteroclinic(self._ys))
        assert jnp.allclose(roundtrip, self._ys, atol=1e-6), (
            f"max |r(H(y)) - y| = {jnp.max(jnp.abs(roundtrip - self._ys)):.2e}"
        )

    def test_H_is_inverse_of_r(self):
        """H(r(u)) = u for u ∈ (0, 2π) (away from endpoints)."""
        us = jnp.linspace(0.05, 2 * jnp.pi - 0.05, 200)
        roundtrip = heteroclinic(r(us))
        assert jnp.allclose(roundtrip, us, atol=1e-6), (
            f"max |H(r(u)) - u| = {jnp.max(jnp.abs(roundtrip - us)):.2e}"
        )

    def test_r_explicit_formula(self):
        """r(u) = log(tan(u/4)) agrees with numerical inversion of H."""
        us = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 100)
        assert jnp.allclose(r(us), jnp.log(jnp.tan(us / 4.0)), atol=1e-7)

    def test_r_grad_finite(self):
        """jax.grad(r) is finite at interior points."""
        grad_r = jax.grad(lambda u_: r(u_))
        for u_val in [0.5, jnp.pi, 5.0]:
            g = grad_r(jnp.array(u_val))
            assert jnp.isfinite(g), f"grad r not finite at u={u_val}: {g}"
