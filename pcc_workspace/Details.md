#  Forward Kinematics details

       z ^
         |
         |   bend-plane angle φ
         |  ↗  b(φ) = [cosφ, sinφ, 0]
         O----> x
        /
       y         rotation axis a(φ) = [-sinφ, cosφ, 0]  (⊥ to b in xy)
       ⟳ θ about a(φ), arc length L, curvature κ = θ/L
         (insertion d is along +z)


## 1) Kinematic parameters and state

For a robot with `n` PCC sections, the per–section parameters are
- **bend angle** \(\theta_i \in \mathbb{R}\) (radians, but applied two typical values in the sample: 90° or 135°, respectively for outer and inner segments),
- **bend-plane angle** \(\phi_i \in [0,2\pi)\) (radians),
- **arc length** \(L_i \ge 0\) (meters),

with curvature \(\kappa_i = \theta_i/L_i\) when \(L_i>0\). There is also a base **insertion (a.k.a. translation)** \(d \in \mathbb{R}\) (meters), modeled as a prismatic joint translating along \(+z\).

The end-effector pose is obtained by serial composition
\[
T_{\mathrm{ee}}(q)=T_{\mathrm{ins}}(d)\,\prod_{i=1}^{n} T_i(\phi_i,\theta_i,L_i),\quad
T_{\mathrm{ins}}(d)=\begin{bmatrix}I_3 & \begin{bmatrix}0\\0\\ d\end{bmatrix}\\ 0 & 1\end{bmatrix}.
\]

---

## 2) One-section transform \(T(\phi,\theta,L)\)

Sections' general homogeneous transform are
\[
T(\phi,\theta,L)=\begin{bmatrix}R(\phi,\theta) & p(\phi,\theta,L)\\ 0 & 1\end{bmatrix}.
\]

which is listed below.

### 2.1 Rotation \(R(\phi,\theta)\) 

Let \([v]_\times\) be the skew-symmetric matrix for the cross product with \(v\). Rotation about the fixed unit axis \(a(\phi)\) by angle \(\theta\) is
\[
R(\phi,\theta)=I_3+\sin\theta\,[a]_\times+(1-\cos\theta)\,[a]_\times^2.
\]

**Explicit 3×3 components**:

Abbreviate \(c_\phi=\cos\phi,\ s_\phi=\sin\phi,\ c_\theta=\cos\theta,\ s_\theta=\sin\theta\), and set \(c^2=c_\phi^2,\ s^2=s_\phi^2,\ sc=s_\phi c_\phi\), 

then we have
\[
R(\phi,\theta)=
\begin{bmatrix}
c^2(c_\theta-1)+1 & sc(c_\theta-1)    & c_\phi s_\theta\\[2pt]
sc(c_\theta-1)    & s^2(c_\theta-1)+c_\theta & s_\phi s_\theta\\[2pt]
-\,c_\phi s_\theta & -\,s_\phi s_\theta & c_\theta
\end{bmatrix}.
\]

### 2.2 Translation \(p(\phi,\theta,L)\)

Using radius \(r=1/\kappa\) and the standard PCC arc in the \(xz\)-plane, then rotating the plane by \(\phi\), the tip translation is
\[
p(\phi,\theta,L)=\frac{1}{\kappa}
\begin{bmatrix}
(1-\cos\theta)\cos\phi\\[2pt]
(1-\cos\theta)\sin\phi\\[2pt]
\sin\theta
\end{bmatrix}.
\]


### 2.3 Small-angle limits

When \(|\theta|\ll 1\),
\[
\sin\theta\approx\theta,\qquad 1-\cos\theta\approx \tfrac{\theta^2}{2},
\]
hence
\[
p(\phi,\theta,L)\approx
\begin{bmatrix}
\tfrac12 L\theta\cos\phi\\[2pt]
\tfrac12 L\theta\sin\phi\\[2pt]
L
\end{bmatrix}.
\]
This is the small-angle approximation used in the code when \(|\theta|<10^{-4}\).

**Numerical guards in the code**
- If \(|\kappa|<\varepsilon\) the code uses `k_safe = eps` to avoid division by zero **and** simultaneously uses the small-angle branch for \(|\theta|<10^{-4}\), which gives the correct limit.
- If a section’s `L < small_L_eps`, the section transform is forced to identity \(T=I_4\) (no rotation and no translation).

---

## 3) Serial composition over sections

Let \(T_i = T(\phi_i,\theta_i,L_i)\). The forward kinematics is
\[
T_{\mathrm{ee}}=T_{\mathrm{ins}}(d)\,T_1 T_2 \cdots T_n,
\]
implemented as left-to-right matrix products in `_compose_fk_from_params`. For each sampled configuration, the code also extracts the tip point as \(p_{\mathrm{ee}}\) (the top-right column of \(T_{\mathrm{ee}}\)).

