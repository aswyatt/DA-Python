# %% [markdown]
# # High Harmonic Generation Dipole Moment in Hydrogen-like Potential
# 
# ## Theory
# 
# ### Lewenstein Model for HHG
# 
# According to the Lewenstein model [1], we can calculate the dipole moment as
# $$\vec{x}(t_r) \simeq 2\mathbb{R}\left\{\int_{t_i}^{t_r} \mathrm{d}t\, 
# \vec{d}^\ast\left(\vec{p}_\text{st} + \vec{A}(t_r)\right)
# e^{-iS(\vec{p}_\text{st}, t_r, t_i)}
# \left[\vec{E}(t_i)\cdot\vec{d}\left(\vec{p}_{st}+\vec{A}(t_i)\right)\right]\right\}.$$
# 
# The transition dipole matrix elements can be calculated using
# $$\vec{d}(\psi_f, \psi_i) = \langle\psi_f|\vec{r}|\psi_i\rangle.$$
# 
# Under the strong-field approximation, we replace the final state as that of a plane wave with asymptotic momentum $\vec{p}$, yielding
# $$\begin{matrix}
# \vec{d}(\vec{p}) & \simeq & \int \mathrm{d}\vec{r}\, e^{-i\vec{p}\cdot\vec{r}}\vec{r}\psi_0(\vec{r})\\
# & = & i\nabla\tilde{\psi_0}(\vec{p})
# \end{matrix}.$$
# Thus we can approximate the transition dipole matrix elements using the gradient of the ground-state orbital in momentum space.
# 
# ### Hydrogen-like Orbitals
# 
# The momentum distribution of a hydrogen-like orbital in spherical co-ordinates is given as [2]
# $$
# \varUpsilon_{nlm}\left(p, \theta,\phi\right) = 
# \frac{\pi 2^{2l+4} l!}{\left(2\pi\gamma\right)^{3/2}}
# \sqrt{\frac{n(n-l-1)!}{(n+l)!}}
# \frac{\zeta^l}{(\zeta^2+1)^{l+2}} C_{n-l-1}^{l+1}\left(\frac{\zeta^2-1}{\zeta^2+1}\right)Z_l^m(\theta, \phi),
# $$
# whereby I have used the real forms of the spherical harmonics:
# 
# $$
# Z_n^m(\theta, \phi) = 
# \begin{cases}
#     \frac{1}{\sqrt{2}} \left[Y_n^m(\theta, \phi) + \overline{Y_n^m}(\theta, \phi)\right] & \text{if } m>0\\
#     Y_n^m(\theta, \phi) & \text{if } m=0\\
#     \frac{1}{i\sqrt{2}}\left[Y_n^m(\theta, \phi) - \overline{Y_n^m}(\theta, \phi)\right] & \text{if } m<0
# \end{cases}
# $$
# 
# with $\overline{Y}$ denoted complex conjugate.
# 
# For s-orbitals, $l=m=0$ and for p-orbitals, $l=1$ and $m=\{-1, 0, 1\}$. For the real orbitals defined above, $m=1$ results in the $np_x$ orbital, $m=-1$ gives the $np_y$ orbital and $m=0$ gives the $np_z$ orbital whereby the subscript is the "axis of revolution" of the radial function.
# 
# ### HHG Source
# 
# The dipole moment can either be found by integrating the temporal integral in the Lewenstein formula, or via stationary phase analysis. In either case, the transition dipole matrix element needs to be evaluated at the time of ionization, $t_i$, and at the time of recombination, $t_r$. For any pair of times, $(t_i, t_r)$, one can evaluate the stationary momentum, $\vec{p}_\text{st}$. Thus we need to calculate the following part of the integrand in the Lewenstein model, the orbital-resolved HHG source term,
# 
# $$
# \vec{S} = \vec{d}_r \left[\vec{E}_i \cdot \vec{d}_i\right]
# $$
# where $i,r$ represent the value at ionization or recombination respectively.
# 
# Additionally, if we choose our co-ordinate axes such that $\hat{z} = \hat{k}$, i.e. $z$ is aligned with the electric field's wave vector, then the electron's trajectory is isolated to the x-y plane (must start and end at the origin). 
# 
# ## Degenerate states
# 
# In the case of degenerate valance orbitals, one needs to calculate the dipole moment for each orbital individually. Since HHG is dominated by the process in which the electron ionizes and recombines from/to the same orbital, phase of the ground state orbital cancels out and the process is coherent. Therefore the dipole moment from each degenerate orbital are coherent from each other. Since the integral is a linear operator, one can calculate the coherent sum of all HHG source terms before performing the integral. Note that this is not the same as performing a coherent sum over transition dipole matrix elements -- it is the full process of ionization and recombination that is coherent and therefore permits a coherent sum over all pathways.
# 
# 1. M. Lewenstein et al., Theory of high-harmonic generation by low-frequency laser fields. Phys. Rev. A. 49, 2117–2132 (1994).
# 1. B. Podolsky, L. Pauling, The momentum distribution in hydrogen-like atoms. Phys. Rev. 34, 109–116 (1929).
# 

# %% [markdown]
# ## Import packages

# %%
# Import sympy package
import sympy as sp

# Import math & numpy packages
import math as mth
import numpy as np

# Ensure output displayed as LaTeX
sp.init_printing(use_unicode=False)

# %% [markdown]
# ## Define symbols
# * $E_i$ is the electric field at time of ionization.
# * $\vec{p}_s = -\int_{t_i}^{t_r} \vec{A}(t^\prime)/(t_r-t_i)\,\text{d}t^\prime$ is the stationary momentum governed by the electric field only (i.e. independent of the atom)
# * $\vec{p}_i = \vec{p}_s + \vec{A}(t_i)$ is the electron momentum at time of ionization (determined by the electric field & independent of the atom).
# * $\vec{p}_r = \vec{p}_s + A(t_r)$ is the electron momentum at time of recombination (determined by the electric field & independent of the atom).
# * $\phi_i$ is the azimuth angle made by the electron momemtum upon ionization.
# * $\phi_r$ is the azimuth angle made by the electron mometum upon recombination.
# * $\Delta = \phi_i - \phi_r$ is the difference in azimuth angles at ionization and recombination.
# * $\alpha$ is the angle made by the electric field at ionization wrt the electron momentum upon ionization.
# * $\tau = \omega t_i$ is the phase of the electric field at the time of ionization.
# * $n,l,m$ are the quantum numbers
# * $p, \theta, \phi$ are spherical coordinates in momentum space:
#  * $p \geq 0$ is the radial component.
#  * $0 \leq \theta \leq \pi$ is the polar angle wrt the z-axis.
#  * $0 \leq \phi < 2\pi$ is the azimuthal angle wrt the x-axis pointing in the direction of the y-axis (right-hand rule).
# * $\gamma = \sqrt{2I_p}$ is the binding momentum parameter of the orbital determined by the ionization potential $I_p$.
# * $\zeta = p/\gamma$, where $p$ is the radial component of the momentum $\vec{p}$.

# %%
# Generic symbols
l, m = sp.symbols("l m", real=True)
alpha, theta, phi, phi_i, phi_r, theta_i, theta_r, alpha, varepsilon, Delta = sp.symbols(
    "alpha theta phi phi_i phi_r theta_i theta_r alpha varepsilon Delta", real=True
)
n, gamma, E_i = sp.symbols("n gamma E_i", real=True, positive=True)
p, p_i, p_r, p_x, p_y, p_z = sp.symbols(
    "p p_i p_r p_x p_y p_z", real=True, positive=True
)
PI = sp.pi
zeta = p / gamma

# Use real spherical harmonics (or not)
REAL = True

# Define polarization in x-y plane
THETA = PI / 2

# Shorthand for trig expressions
sin_theta = sp.sin(theta)
cos_theta = sp.cos(theta)
sin_phi = sp.sin(phi)
cos_phi = sp.cos(phi)

# %% [markdown]
# ## Define rotation matrices

# %% [markdown]
# ### Vector rotation matrix spherical --> Cartesian
# $$
# \begin{bmatrix}
# \mathbf{\hat{x}}\\ 
# \mathbf{\hat{y}}\\ 
# \mathbf{\hat{z}}
# \end{bmatrix}
#  = \begin{bmatrix}
# \sin\theta\cos\phi & \cos\theta\cos\phi & -\sin\phi\\
# \sin\theta\sin\phi & \cos\theta\sin\phi & \cos\phi\\
# \cos\theta & -\sin\theta & 0
# \end{bmatrix}
# = \texttt{R}(\theta, \phi)\begin{bmatrix}
# \mathbf{\hat{p}}\\ 
# \mathbf{\hat{\theta}}\\ 
# \mathbf{\hat{\phi}}
# \end{bmatrix}
# $$

# %%
spherical_to_cartesian_matrix = sp.Matrix(
    [
        [sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi],
        [sin_theta * sin_phi, cos_theta * sin_phi, cos_phi],
        [cos_theta, -sin_theta, 0],
    ]
)

# %% [markdown]
# ### Rotation matrices
# $$
# \texttt{R}_x(\theta) =
# \begin{bmatrix}
# 1 & 0 & 0\\
# 0 & \cos\theta & -\sin\theta\\
# 0 & \sin\theta & \cos\theta
# \end{bmatrix}
# \\
# \texttt{R}_y(\theta) =
# \begin{bmatrix}
# \cos\theta & 0 & \sin\theta\\
# 0 & 1 & 0\\
# -\sin\theta & 0 & \cos\theta\\
# \end{bmatrix}
# \\
# \texttt{R}_z(\phi) =
# \begin{bmatrix}
# \cos\phi & -\sin\phi & 0\\
# \sin\phi & \cos\phi & 0\\
# 0 & 0 & 1
# \end{bmatrix}\;
# $$

# %%
print("R_x = ")
sp.pprint(Rx := sp.Matrix([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]))

print("\nR_y = ")
sp.pprint(Ry := sp.Matrix([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]))

print("\nR_x = ")
sp.pprint(Rz := sp.Matrix([[cos_phi, -sin_phi, 0], [sin_phi, cos_phi, 0], [0, 0, 1]]))

# %% [markdown]
# ### Convert from spherical to Cartesian co-ordinates
# $$
# p_x = p \sin\theta \cos\phi \\
# p_y = p \sin\theta \sin\phi \\
# p_z = p \cos\theta
# $$

# %% [markdown]
# ### Substitute values for spherical coordinates

# %%
def vector_subst(expr, P, th, ph):
    return sp.trigsimp(expr.subs([(p, P), (theta, th), (phi, ph)]))

# %% [markdown]
# ### Simplification of vector

# %%
def vector_operation(V, *args):
    v_out = []
    for v in V:
        for op in args:
            v = op(v)
        v_out.append(v)
    return sp.Matrix(v_out)


def vector_collect(V, *args):
    return vector_operation(V, lambda x: sp.collect(x, args))

# %% [markdown]
# ## Define electric 
# Electric field is defined in the x-y plane
# $$
# E 
# = \begin{bmatrix}E_x\\E_y\\0\end{bmatrix}
# = \frac{E_0}{\sqrt{1+\varepsilon^2}} \begin{bmatrix}\cos(\omega t)\\\varepsilon\sin(\omega t)\\0\end{bmatrix}
# = E_0^\prime \begin{bmatrix}\cos(\tau)\\\varepsilon\sin(\tau)\\0\end{bmatrix}
# $$
# 
# where $\tau = \omega t_i$
# 
# However, the electric field only enters the Lewenstein model at the time of ionisation. We can then define this field as
# $$
# \vec{E}_i = E_i 
# \begin{bmatrix}
#     \cos(\alpha + \phi_i)\\
#     \sin(\alpha + \phi_i)\\
#     0
# \end{bmatrix}
# $$

# %%
print("\nE_i = ")
sp.pprint(E := E_i*sp.Matrix([sp.cos(alpha + phi_i), sp.sin(alpha + phi_i), 0]))

# %% [markdown]
# # Hydrogen-like orbitals

# %% [markdown]
# ## Definitions

# %% [markdown]
# ### Redefine spherical harmonics
# Name of function is unintuitive and requires additional angle arguments, so I redefine and simplify the calling convention

# %%
# Redefine real spherical harmonics
def Zlm(l, m):
    return sp.Znm(l, m, theta, phi)
Zlm(l, m)

# %%
# Redefine complex spherical harmonics
def Ylm(l, m):
    return sp.Ynm(l, m, theta, phi)
Ylm(l,m)

# %% [markdown]
# ### Define hydrogen-like orbitals in momentum space
# $$
# \varUpsilon_{nlm}\left(p, \theta,\phi\right) = 
# \frac{\pi 2^{2l+4} l!}{\left(2\pi\gamma\right)^{3/2}}
# \sqrt{\frac{n(n-l-1)!}{(n+l)!}}
# \frac{\zeta^l}{(\zeta^2+1)^{l+2}} C_{n-l-1}^{l+1}\left(\frac{\zeta^2-1}{\zeta^2+1}\right)Z_l^m(\theta, \phi)
# $$

# %%
def wavefunction(n, l, m):
    zeta2 = zeta**2
    A = PI * 2**(2*l+4) * sp.factorial(l) / (2*PI*gamma)**sp.Rational(3,2)
    B = sp.sqrt(n*sp.factorial(n-l-1)/sp.factorial(n+l))
    C = (zeta**l/(zeta2+1)**(l+2)) * sp.gegenbauer(n-l-1, l+1, (zeta2-1)/(zeta2+1))
    if REAL:
        U = Zlm(l, m)
    else:
        U = Ylm(l, m)       
    return U * A * B * C

print("\nupsilon(n,l,m) = ")
sp.pprint(wavefunction(n, l, m))

# %% [markdown]
# ### Define dipole transition matrix element
# Vector is in spherical coordinates with $\hat{p}=\frac{\vec{p}}{\left| \vec{p} \right|}$ in the direction of the momentum.
# $$
# \vec{d}_p(\vec{p}) = 
# i\begin{bmatrix}
# \partial_p \varUpsilon(p, \theta, \phi)\\
# (1/p)\partial_\theta \varUpsilon(p, \theta, \phi)\\
# (1/p\sin\theta)\partial_\phi \varUpsilon(p, \theta, \phi)\\
# \end{bmatrix}
# $$

# %%
def transition_dipole_spherical(upsilon):
    dp = sp.simplify(sp.diff(upsilon, p))
    dtheta = sp.simplify(sp.diff(upsilon, theta)/p)
    dphi = sp.simplify(sp.diff(upsilon, phi)/(p*sp.sin(theta)))
    expr = sp.Matrix([e.factor().collect(p) for e in [dp, dtheta, dphi]])
    return sp.I*expr

# %%
def spherical_to_cartesian(spherical_vector):
    rules = {
        cos_phi: p_x / (p * sin_theta),
        sin_phi: p_y / (p * sin_theta),
        sin_theta: sp.sqrt(1 - cos_theta**2),
        sin_theta * cos_phi: p_x / p,
        sin_theta * sin_phi: p_y / p,
        cos_theta: p_z / p,
        p_z: sp.sqrt(p**2 - p_x**2 - p_y**2),
    }
    expr = (spherical_to_cartesian_matrix * spherical_vector).subs(rules)
    return vector_operation(sp.simplify(expr), sp.factor)

# %% [markdown]
# ### Define dipole moment
# Vector is in spherical coordinates with $\hat{p} = \vec{p}_r/\left|\vec{p}_r\right|$ in the direction of the momentum at time of recombination, since $\vec{E}.{d}$ is a scalar,
# $$
# \vec{x} \propto 
# i\int_{-\infty}^{t_r} \vec{d}_p^\ast(\vec{p}_r) e^{-iS}\left[\vec{E}_i\cdot\vec{d}_p(\vec{p}_i)\right] \, \text{d}t_i + \text{c.c.}
# $$
# 
# <!--
# $$\vec{\widetilde{d}_p}(x, y, z) = \texttt{R}_z\left(-\phi_i-\phi_r\right)\texttt{R} \left(\pi/2, \phi_i+\phi_r\right) \vec{d}_p\left(p, \pi/2, \phi_i+\phi_r\right)$$
# -->
# 
# The dipole moment is defined relative to the momentum vector at time of recombination, not to the quantization axes which are arbitrary. We however want to find the dipole moment wrt our lab-frame co-ordinate system, which is defined by the electric field:
# 1. Define dipole moment in vector spherical coordinates
# 1. Rotate ionisation transition dipole matrix to Cartesian coordinates
#    * $p_\text{st}$ has radial magnitude $p_i$ at angle $\phi_i$ wrt $\hat{x}$.
# 1. Dot with electric field at $\tau$ in Cartesian co-ordinates (scalar)
# 1. Define recombination transition dipole matrix
#    * $p_\text{st}$ has radial magnitude $p_r$ at angle $\phi_i + \phi_r$ wrt $\hat{x}$.
# 1. Multiply together to yield net dipole moment in spherical co-ordinates
# 
# 
# As the ellipticity changes, so will the values of $\alpha$ and $\phi_r$, as well as $E_i$, $p_i$ and $p_r$.

# %%
def hhg_source_cartesian(upsilon, E):
    # Dipole transition matrix element in vector Cartesian coordinates
    d = spherical_to_cartesian_matrix * transition_dipole_spherical(upsilon)

    # Ionisation dipole transition matrix element in Cartesian coordinates
    di = d.subs({p: p_i, theta: theta_i, phi: phi_i})

    # Net ionisation rate
    dI = E.T * di

    # Recombination dipole transition matrix element in spherical coordinates
    dr = d.subs({p: p_r, theta: theta_r, phi: phi_r})

    # Simplify
    expr = sp.simplify(vector_collect(sp.trigsimp(dr * dI), p_r**2, p_i**2, gamma**2))

    # Return dipole moment
    return expr


def ratio(S):
    Sx = S[0, 0]
    Sy = S[1, 0]
    S_par = sp.simplify(Sx * sp.cos(phi_r) + Sy * sp.sin(phi_r))
    S_perp = sp.simplify(-Sx * sp.sin(phi_r) + Sy * sp.cos(phi_r))
    r = (
        (S_perp / S_par)
        .simplify()
        .factor()
        .subs({phi_i - phi_r: Delta})
        .simplify()
        .factor()
    )
    return r


# Apply physical constraint theta_i = theta_r = pi/2 because E in x-y plane
def physical_constraint(S):
    return S.subs({theta_i: THETA, theta_r: THETA}).simplify().trigsimp()

# %% [markdown]
# ## HHG source term ratios
# 
# Calculates the ratio between the components of the net HHG source term parallel/perpendicular with respect to the recombination momentum vector.

# %% [markdown]
# ### 1s Orbital
# 
# Check that this yields 0 (transition dipole moment is purely radial, so no perpendicular component)

# %%
U1s = sp.expand_func(wavefunction(1, 0, 0)).simplify()
S1s = hhg_source_cartesian(U1s, E)
print("\nR1s = ")
sp.pprint(R1s := ratio(physical_constraint(S1s)))

# %% [markdown]
# ### 2p Orbitals

# %%
U2px = sp.expand_func(wavefunction(2, 1, 1)).simplify()
U2py = sp.expand_func(wavefunction(2, 1, -1)).simplify()
U2pz = sp.expand_func(wavefunction(2, 1, 0)).simplify()

S2px = hhg_source_cartesian(U2px, E)
S2py = hhg_source_cartesian(U2py, E)
S2pz = hhg_source_cartesian(U2pz, E)
S2p = S2px + S2py + S2pz

print("\nR2p = ")
sp.pprint(R2p := ratio(physical_constraint(S2p)))
print(f"\n{sp.latex(R2p)}")
