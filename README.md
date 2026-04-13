# Explicit solutions to the elliptic sine-gordon on $\mathbb{R}^2$

JAX implementation of the explicit 2*n*-ended solutions to the elliptic sine-Gordon equation

$$\Delta u = \sin u, \qquad u : \mathbb{R}^2 \to (0, 2\pi),$$

constructed via the Hirota direct method. 

## References

**[LW]** Y. Liu and J. Wei, Classification of finite Morse index solutions to the elliptic sine-Gordon equation in the plane. Rev. Mat. Iberoam. 38 (2022), no. 2, 355–432.

**[CM]** O. Chodosh and C. Mantoulidis, The p-widths of a surface. Publ. Math. Inst. Hautes Études Sci. 137 (2023), 245–342.


---

## The PDE and solution family

The **elliptic sine-Gordon equation** on the plane is taken to be

$$\Delta u = \sin u = W'(u),$$

where $W(t) = 1 - \cos t$ is the **double-well potential** with wells at $t \in 2\pi\mathbb{Z}$.

For each integer $n \geq 1$, there is a family of **2*n*-ended solutions** $U_n$ parameterised by

- unit vectors $(p_j, q_j) \in S^1$, $j = 1, \dots, n$ — the *end directions*,
- phase offsets $\eta_j^0 \in \mathbb{R}$, $j = 1, \dots, n$.

---


### Hirota tau-function formula

Define the phase variables

$$\eta_j(x,y) = p_j x + q_j y + \eta_j^0,$$

and the pairwise interaction coefficients

$$\alpha(j,k) = \frac{(p_j - p_k)^2 + (q_j - q_k)^2}{(p_j + p_k)^2 + (q_j + q_k)^2}.$$

For each subset $S \subseteq \{1,\dots,n\}$, set

$$A(S) = \prod_{\substack{j < k \\ j,k \in S}} \alpha(j,k), \qquad A(\varnothing) = 1.$$

The tau functions are

$$f_n = \sum_{\substack{S \subseteq \{1,\dots,n\} \\ |S| \text{ even}}} A(S)\, e^{\sum_{j \in S} \eta_j}, \qquad g_n = \sum_{\substack{S \subseteq \{1,\dots,n\} \\ |S| \text{ odd}}} A(S)\, e^{\sum_{j \in S} \eta_j}.$$

The solution is

$$U_n(x, y) = 4 \arctan\!\left(\frac{g_n}{f_n}\right) \in (0, 2\pi).$$

This takes values in $(0, 2\pi)$ and satisfies $\Delta U_n = \sin U_n$.

> **Convention note.** The formula in Liu–Wei uses the opposite sign convention $-\Delta u = \sin u$ and produces solutions in $(-\pi, \pi)$. The formula above is shifted by $\pi$: $U_n^{\text{here}} = U_n^{\text{LW}} + \pi$.

---

## Special cases

### $k = 1$: the heteroclinic (2-ended solution)

With $n=1$, $p_1 = 0$, $q_1 = 1$, $\eta_1^0 = 0$, the formula reduces to the **heteroclinic** (or kink) solution

$$H(y) = 4 \arctan\!\bigl(e^{y}\bigr).$$

This is a function of $y$ only (constant along the $x$-axis), with $H(0) = \pi$, $H \to 0$ as $y \to -\infty$, and $H \to 2\pi$ as $y \to +\infty$. Its inverse is explicit:

$$\mathrm{heteroclinic\_inverse}(u) = H^{-1}(u) = \log\!\tan\!\frac{u}{4}, \qquad u \in (0, 2\pi).$$

### $k = 2$: 4-ended solutions

**Saddle solution** (parameters $p_1 = p_2 = 1/\sqrt{2}$, $q_1 = -q_2 = 1/\sqrt{2}$, $\eta^0 = 0$):

$$U_2(x,y) = 4\arctan\!\left(\frac{\cosh(y/\sqrt{2})}{\cosh(x/\sqrt{2})}\right).$$

**Family $\varphi_{p,q}$** (parameters $p_1 = p_2 = p$, $q_1 = -q_2 = q$ with $p^2 + q^2 = 1$, $\eta_1^0 = \eta_2^0 = \log(p/q)$):

$$\varphi_{p,q}(x,y) = 4\arctan\!\left(\frac{p\cosh(qy)}{q\cosh(px)}\right).$$

As $p/q$ varies, this interpolates between the saddle ($p = q$) and degenerate limits.

---

## Repository structure

```
sine_gordon/
    __init__.py        # public API
    solutions.py       # all implementations
tests/
    test_solutions.py  # pytest suite (closed forms, PDE residual, differentiability, JIT)
pyproject.toml
```

### Public API (`from sine_gordon import ...`)

| Symbol | Description |
|---|---|
| `u_n(p, q, eta0, x, y)` | 2*n*-ended solution; `p`, `q`, `eta0` are `(n,)` arrays |
| `u_n_from_angles(theta, eta0, x, y)` | Same, parameterised by angles $\theta_j$ (unconstrained optimisation) |
| `heteroclinic(y)` | $H(y) = 4\arctan(e^y)$ |
| `heteroclinic_inverse(u)` | $H^{-1}(u) = \log\tan(u/4)$, the inverse of $H$ |
| `double_well(t)` | Double-well potential $W(t) = 1 - \cos t$; satisfies $W'(t) = \sin t$ |
| `alpha_matrix(p, q)` | $(n \times n)$ interaction coefficient matrix |

All functions are JAX-compatible: supports `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.hessian`.

---

## Google Colab quickstart

```python
# Install (run once)
!pip install -q jax jaxlib
!pip install -q git+https://github.com/ochodosh/sine_gordon.git

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sine_gordon import double_well, u_n, u_n_from_angles, heteroclinic, heteroclinic_inverse

# --- Heteroclinic (k=1) ---
ys = jnp.linspace(-5, 5, 300)
H = heteroclinic(ys)          # shape (300,), values in (0, 2pi)

# --- Saddle solution (k=2) ---
s = 1.0 / jnp.sqrt(2.0)
p = jnp.array([s, s])
q = jnp.array([s, -s])
eta0 = jnp.zeros(2)

xs = jnp.linspace(-4, 4, 200)
X, Y = jnp.meshgrid(xs, xs)
U = u_n(p, q, eta0, X, Y)     # shape (200, 200)

# --- Angle parameterisation (convenient for optimisation) ---
theta = jnp.array([jnp.pi / 6, jnp.pi / 3])
U2 = u_n_from_angles(theta, eta0, X, Y)

# --- Gradients w.r.t. parameters ---
loss = lambda th: jnp.mean(u_n_from_angles(th, eta0, X, Y))
grad = jax.grad(loss)(theta)

# --- JIT for speed ---
u_jit = jax.jit(u_n)
U_fast = u_jit(p, q, eta0, X, Y)

# --- Quick plot ---
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ys, H / jnp.pi)
axes[0].set(title=r"Heteroclinic $H(y)/\pi$", xlabel="y")
axes[1].contourf(X, Y, U / jnp.pi, levels=40, cmap="RdBu_r")
axes[1].set(title=r"Saddle $U_2/\pi$", aspect="equal")
plt.colorbar(axes[1].collections[0], ax=axes[1])
plt.tight_layout()
plt.show()
```
