### 1.1 Kinematic parameters and state

For a robot with \(n\) PCC sections, each section \(i\) uses

- **bend angle** \(\theta_i \in \mathbb{R}\) (radians),
- **bend‑plane angle** \(\phi_i \in [0,2\pi)\) (radians),
- **arc length** \(L_i \ge 0\) (meters),

with curvature \(\kappa_i=\theta_i/L_i\) when \(L_i>0\). The base **insertion** (prismatic motion along \(+z\)) is \(d\in\mathbb{R}\) (meters).

The end‑effector pose is obtained by serial composition
\[
T_{\mathrm{ee}}(q)
= T_{\mathrm{ins}}(d)\,\prod_{i=1}^{n} T_i(\phi_i,\theta_i,L_i),
\qquad
T_{\mathrm{ins}}(d)=\begin{bmatrix}I_3 & \begin{bmatrix}0\\0\\ d\end{bmatrix}\\ 0 & 1\end{bmatrix}.
\]

> **Code mapping** — The FK sets `T[:,2,3] = d` for insertion, then multiplies section transforms on the right.

---

### 1.2 One‑section transform \(T(\phi,\theta,L)\)

We write
\[
T(\phi,\theta,L)=
\begin{bmatrix}
R(\phi,\theta) & p(\phi,\theta,L)\\
0 & 1
\end{bmatrix}.
\]

#### Rotation \(R(\phi,\theta)\) via Rodrigues

**Axis and in‑plane direction.**
\(
b(\phi)=[\cos\phi,\ \sin\phi,\ 0]^T,\
a(\phi)=[-\sin\phi,\ \cos\phi,\ 0]^T.
\)

With \(c_\phi=\cos\phi,\, s_\phi=\sin\phi,\, c_\theta=\cos\theta,\, s_\theta=\sin\theta\):
\[
R(\phi,\theta)=
\begin{bmatrix}
c_\phi^2(c_\theta-1)+1 & s_\phi c_\phi(c_\theta-1)    & c_\phi s_\theta\\
s_\phi c_\phi(c_\theta-1)    & s_\phi^2(c_\theta-1)+c_\theta & s_\phi s_\theta\\
-\,c_\phi s_\theta & -\,s_\phi s_\theta & c_\theta
\end{bmatrix}.
\]

#### Translation \(p(\phi,\theta,L)\) from a planar arc

Let \(r=1/\kappa=L/\theta\). For \(\phi=0\) (bending about +y), \(p_0=[r(1-\cos\theta),\, 0,\, r\sin\theta]^T\). Rotate the plane by \(R_z(\phi)\):
\[
p(\phi,\theta,L) \;=\; \frac{1}{\kappa}
\begin{bmatrix}
(1-\cos\theta)\cos\phi\\[2pt]
(1-\cos\theta)\sin\phi\\[2pt]
\sin\theta
\end{bmatrix}.
\]

#### Small‑angle limits (\(|\theta|\ll 1\))

Using \(\sin\theta\!\approx\!\theta,\ 1-\cos\theta\!\approx\!\tfrac{\theta^2}{2}\) and \(r=L/\theta\):
\[
p \approx \begin{bmatrix}\tfrac12 L\theta\cos\phi\\[2pt]\tfrac12 L\theta\sin\phi\\[2pt]L\end{bmatrix},\qquad R\approx I+\theta[a]_\times.
\]

> **Implementation** — For \(|\theta|\) below a tiny threshold, the code switches to this limit so a *straight* section still advances by \(+L\,\hat z\). Passive stubs are pure translations applied **before** the active arc of that section.

---

## 2) Workspace sampling (FK sampler)

We provide a batched FK sampler (CPU NumPy / GPU CuPy) that draws random joint states within `SegmentSpec` bounds and returns point cloud `pts` and full transforms `Ts`:

```python
from pcc_fk import PCCFKSolver  # your module path

fk = PCCFKSolver([outer, inner], translation=None)
pts, Ts = fk.sample_workspace(n_samples=200_000)  # pts: (N,3), Ts: (N,4,4)
```

- **Passive/active split**: each section uses `L_passive` (straight) + `L_active` (bending with θ).  
- **No roll**: only bend‑plane yaw \(\phi\) and bend angle \(\theta\); any `phi_coupling` is enforced (e.g., `lock_prev`, constant offset).  
- **Straight‑segment fix**: if \(L_{\rm act}\to0\) or \(|\theta|\to0\), the section reduces to straight translation by \(L_{\rm act}\).

---
## A) Constraints (mathematical form)

Let a sampled pose be \(T_k\in SE(3)\) with rotation \(R_k\) and tip **approach direction**
\(\displaystyle u_k = R_k\,e_z \in \mathbb{S}^2\) where \(e_z=[0,0,1]^T\).
Let the **bevel direction** in the tip frame be \(b_0=[\sin\alpha,\,0,\,\cos\alpha]^T\) (bevel angle \(\alpha\) w.r.t. \(+z\)). In the world,
\(\displaystyle b_k = R_k\,b_0\).

We use a unit **target normal** \(\hat n(x)\) that may be global (constant \(\hat n_\star\)) or a field defined per position \(x\). In examples below we use a fixed \(\hat n_\star\).

> All constraints are Boolean predicates \(c_i(k)\in\{0,1\}\) evaluated per sample. A sample is **admissible** iff \(\prod_i c_i(k)=1\).

### C1) **No roll about tool axis**
- Already built into the FK/IK model (no axial torsion state).  
- **Impact on dexterity:** The orientation state space excludes axial roll; our dexterity uses only \(u_k\) (approach direction), so C1 **does not change** the definition; it is implicitly enforced by the sampler.

### C2) **Strict bevel-normal alignment**
\[
c_{\text{bevel}}(k) \;=\; \mathbf 1\big[\, b_k\cdot \hat n_\star \;\ge\; \cos(\varepsilon_{\text{bevel}})\,\big].
\]
- Tightening \(\varepsilon_{\text{bevel}}\) **reduces** admissible orientations \(\Rightarrow\) lower coverage.

### C3) **Inner bending-plane coplanarity with a world vector \(v\)**
Let \(n^{(2)}_k\) be the **unit normal** of the inner-segment bending plane, expressed in world. The coplanarity with a world vector \(v\) (e.g., \(\hat n_\star\) or the bevel plane normal) is
\[
c_{\text{plane}}(k) \;=\; \mathbf 1\big[\, |\,n^{(2)}_k \cdot \hat v\,| \;\ge\; \cos(\varepsilon_{\text{plane}})\,\big].
\]
- In our IK, \(n^{(2)}_k\) comes from \(R_{\text{pre},2}\) and \(\phi_2\):  
  \(n^{(2)}_k = R_{\text{pre},2} \,[-\sin\phi_2,\,\cos\phi_2,\,0]^T\).
- **Impact:** Rejects orientations whose **bending plane** misaligns from the target plane \(\Rightarrow\) fewer bins in each cell.

> If \(n^{(2)}_k\) is not available in your FK sampling, instrument the sampler to optionally return it per sample (extras array). For unconstrained FK visualization, this mask can be omitted.

### C4) **Contact-angle band (tip-axis vs. surface normal)**
Let \(\beta_k=\arccos(u_k\cdot \hat n_\star)\). Require
\[
c_{\text{angle}}(k) \;=\; \mathbf 1\big[\, |\beta_k-\beta_{\text{req}}| \le \varepsilon_{\text{ang}} \,\big].
\]
- **Impact:** Cuts away orientations outside the permitted contact band.

### C5) **Front-side constraint**
\[
c_{\text{front}}(k) \;=\; \mathbf 1\big[\, u_k\cdot \hat n_\star \;\ge\; \cos(\beta_{\text{front}})\,\big].
\]
- With \(\beta_{\text{front}}=90^\circ\) this is simply \(u_k\cdot \hat n_\star \ge 0\) (rejects back-facing approaches).

### C6) **Segment bounds (mechanical limits)**
Already enforced during FK sampling (lengths, \(\theta\) ranges, \(\phi\) windows, passive stubs). These **shape** the reachable workspace and hence dexterity indirectly.

---

## 3) Dexterity metric (orientation coverage)

Given sampled poses \(T_k\) with rotation \(R_k\), define the approach direction
\(
u_k = R_k e_z\in\mathbb{S}^2,\ e_z=[0,0,1]^T.
\)
Discretize the sphere by **azimuth–elevation** bins `bins_az_el=(n_az,n_el)`:
\[
\operatorname{az}(u)=\text{atan2}(u_y,u_x)\in[0,2\pi),\quad
\operatorname{el}(u)=\arcsin(u_z)\in[-\tfrac{\pi}{2},\tfrac{\pi}{2}].
\]
The bin index is
\[
b(u)=\Big\lfloor\tfrac{\operatorname{az}(u)}{2\pi}\,n_{az}\Big\rfloor \;+\;
n_{az}\Big\lfloor\tfrac{\operatorname{el}(u)+\pi/2}{\pi}\,n_{el}\Big\rfloor,
\]
and \(B=n_{az}n_{el}\) is the total number of orientation bins.

For a **voxel** (3D) or **grid cell** (2D slice), let \(S\) be the indices whose end‑effector position lies inside the cell. The **dexterity** (orientation coverage) is
\[
\delta \;=\; \frac{\big|\{\,b(u_k): k\in S\,\}\big|}{B}\in[0,1].
\]
Cells/voxels with fewer than a threshold number of samples are marked `NaN` to avoid noisy estimates.
