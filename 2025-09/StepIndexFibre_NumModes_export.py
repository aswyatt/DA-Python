# %% [markdown]
# # Step-Index Fibre
# 
# The number of modes that a fibre can support can be approximated by the expression
# $$
# N_m \approx \frac{V^2}{2}.
# $$
# where
# $$
# V = a k_0 \text{NA}.
# $$
# 
# However, a complete analysis is required to obtain the exact number.
# 
# ## Analytic Derivation
# 
# The fibre is subject to the refractive index profile
# $$
# n(r\leq a) = n_\mathrm{core}\\
# n(r > a) = n_\mathrm{clad}\\
# $$
# 
# and field subject to the boundary conditions
# $$
# E(r^{(-)}=a) = E(r^{(+)}=a)\\
# E^\prime(r^{(-)}=a) = E^\prime(r^{(+)}=a)
# $$
# where $r^{(-)}$ and $r^{(+)}$ refer to the radial co-ordinate when approacing from $-\infty$ and $+\infty$ respectively.
# 
# The solution is
# $$
# E(r,\phi,z,t) = \sum_{lm} \psi_{lm}(r) e^{il\phi} e^{i\beta_{lm} z} e^{-i\omega t} \left|a_{lm}\right|e^{i\theta_{lm}}
# $$
# 
# with
# $$
# \psi_{lm}(r) = \begin{cases}
# A_{lm} J_l\left(u_{lm}\frac{r}{a}\right) & r\leq a\\
# A_{lm} \frac{J_l\left(u_{lm}\right)}{K_l\left(w_{lm}\right)} K_l\left(w_{lm}\frac{r}{a}\right) & r\geq a
# \end{cases}
# $$
# 
# where 
# $$
# u_{lm} = a\sqrt{n_\mathrm{core}^2k_0^2 - \beta_{lm}^2}\\
# w_{lm} = a\sqrt{\beta_{lm}^2 - n_\mathrm{clad}^2k_0^2}
# $$ 
# where $J_l$ is the Bessel function of the _1<sup>st</sup>_ kind, and $K_l$ is the modified Bessel function of the _2<sup>nd</sup>_ kind.
# 
# For the mode to be propagating, $u_{lm}$ and $w_{lm}$ must be real, thus
# $$
# n_\mathrm{clad} \leq \beta_{lm}/k_0 \leq n_\mathrm{core}
# $$ 
# 
# From this, taking the negative signs, one can conclude
# $$
# j_{l-1,m} < V < j_{l,m}
# $$
# where the roots of the Bessel function are labelled $j_{lm}$, i.e.$J_l(j_{lm}) = 0$.
# 
# **Note that for $l=0$, there is also an additional solution for the fundamental mode**

# %%
import numpy as np
from scipy import special

PI = np.pi
TAU = 2*PI

# %% [markdown]
# ## Functions

# %%
# Refractive index of bulk fused silica (SiO2) - Malitson (1965)
def ref_index_fused_silica(l_um):
    return np.sqrt(
        1
        + 0.6961663 / (1 - (0.0684043 / l_um) ** 2)
        + 0.4079426 / (1 - (0.1162414 / l_um) ** 2)
        + 0.8974794 / (1 - (9.896161 / l_um) ** 2)
    )

# Calculate the maximum number of modes (propagation constant must be real)
#  u_max = V
def num_modes(V: float, l: int = 0, Nm: int = 10):
    j_lm = lambda Nm: special.jn_zeros(l-1, Nm)
    Nm = max(Nm, 1)
    roots_lm1 = j_lm(Nm)
    while roots_lm1[-1] < V:
        Nm *= 2
        roots_lm1 = j_lm(Nm)
    # For l=0, include the first root larger than V
    return sum(roots_lm1 < V).item() + (l==0)

# %% [markdown]
# ## Define Fibre Parameters

# %%
l0 = 633e-3 # Wavelength [um]
k0 = TAU/l0
d = 10 # Core diameter [um]
a = d/2 # Fibre radius [um]
NA = 0.12 # Numerical aperture

n_clad = ref_index_fused_silica(l0) # Ref. index of cladding
n_core = np.sqrt(NA**2 + n_clad**2) # Ref. index of core

# Normalized frequency
V = a * k0 * NA

# %% [markdown]
# ## Find number of modes
# 
# For a given $l$, find number of roots, $J_l(\rho_{lm})=0$ such that $\rho_{lm} \leq V$.
# 
# For $l=0$, there are 2 modes per root ()
# For $l\geq1$, there are 4 modes per root ($\pm l$ and two polarizations).

# %%
N_lm = []
l = 0
while True:
    if (Nm := num_modes(V, l))==0:
        break
    N_lm.append(Nm)
    l += 1
print(N_lm, 2*N_lm[0] + 4*sum(N_lm[1:]))

# %% [markdown]
# ### Check Modes
# 
# Output the zeros of the Bessel functions to make sure we haven't missed any. The first non-guided root for each l, as well as the first non-guided l is displayed to check it is larger than V.

# %%
print(f"V = {V:.6f}")
Nl = len(N_lm)
for l in range(Nl+1):
    Nm = N_lm[l] if l<Nl else 0
    print(f"{l = }, {Nm=}:", special.jn_zeros(l-1, Nm+1))

# %% [markdown]
# ## Display Results

# %%
print("Valid LP modes:")
_ = [print(f"LP_{l}{m}") for l, N_l in enumerate(N_lm) for m in range(N_l)]
print(f"V = {V:.3f}")
print(f"Approximate number of modes (V^2/2): {V**2/2:.2f}")

# Two polarizations for l=0, four for l>0 
print("\nTotal number of modes:", 2*N_lm[0] + 4*sum(N_lm[1:]))


