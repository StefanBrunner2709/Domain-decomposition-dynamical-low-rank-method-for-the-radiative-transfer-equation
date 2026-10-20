# Infos to d1x1

> Goal is to solve the radiative transfer equation with a dynamical low-rank (DLR) method, with and without domain decomposition (DD)

Equation

$$\partial_t f(t,x,\mu) + \frac{1}{\epsilon} \mu \partial_x f(t,x,\mu) = \frac{1}{\epsilon^2} (\frac{1}{2} \int_{-1}^{1} f(t,x,\mu') d\mu' - f(t,x,\mu))$$

+ $f(t,x,\mu)$ is the particle density at position $x$ in flying direction $\mu$
+ $\mu = \cos(\theta) \in [-1,1]$ is the flight direction $\mu>0$ means right, $\mu<0$ means left
+ $\mu \partial_x f(t,x,\mu)$ is the transport term
+ $\epsilon$ is the free path, if $\epsilon$ is small, there is a lot of scattering and the solution behaves like difffusion.

**Boundary conditions (inflow)**

$f(t,0,\mu) = \tanh(t)$ for $\mu>0$, $f(t,1,\mu) = \tanh(t)$ for $\mu<0$

In the outgoing directions, nothing is prescribed the code extrapolates linearly from the interior, $f[0] - (f[1] - f[0])$

**Initial condition**

$f(0, x, \mu) = 0$, which means physically that radiation is swithec on at time $t=0$ and ramps up as $\tanh(t)$ at the boundaries.

**Discretization**

+ $x$: `Nx`interior points with spacing `dx`. $0$ and $1$ are not on the grid and stored as ghost values in `F_b`
+ $\mu$: `Nmu`points in [-1,1] with spacing `dmu` 
+ $f$ is then a matrix $f \in \mathbb{R}^{Nx \times Nmu}$, with $f[i,j] = f(t, x_i, \mu_j)$ 