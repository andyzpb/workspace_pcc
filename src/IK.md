# Two-Segment PCC IK (Closed-Form + Search) — Implementation Notes

*This document is an exact specification of the inverse kinematics implemented in the provided Python code. Wording and equations match the implementation, prioritising numerical stability and bound handling as coded. Units are SI; angles are **radians** unless otherwise noted. Spelling follows British English.*

---

## 0) Problem statement

Given a target contact point \(P_\star\in\mathbb{R}^3\) and a target unit surface normal \(\hat{\boldsymbol n}_\star\in\mathbb{S}^2\), find a feasible configuration for a **two-segment** piecewise constant-curvature (PCC) robot:
- **Outer** segment (index 1 in the maths, but stored as `seg1` in the solver).
- **Inner** segment (index 2, stored as `seg2`).

A scalar **base feed** \(d\) acts along world \(+\hat{\boldsymbol z}\) and **affects position only**. The solver is formulated by moving \(d\) to the target side:
\[
P_\star^{(d)}:=P_\star-d\,\hat{\boldsymbol z}.
\]

A configuration is **acceptable** iff
\[
\|p_{\text{tip}}^{(d)}-P_\star\|\le \varepsilon_{\text{pos}},\qquad
\angle(\boldsymbol b_{\text{world}},\hat{\boldsymbol n}_\star)\le \varepsilon_{\text{bevel}},
\]
where \(p_{\text{tip}}^{(d)}=p_{\text{tip}}+d\,\hat{\boldsymbol z}\).

The **bevel** direction is
\[
\boldsymbol b_{\text{world}}=R_{\text{tip}}\boldsymbol b_0,\qquad 
\boldsymbol b_0=\begin{bmatrix}\sin\alpha\\[2pt]0\\[2pt]\cos\alpha\end{bmatrix},
\]
with known bevel angle \(\alpha\) (field `bevel_angle_deg`).

---

## 1) PCC kinematics used by the solver

### 1.1 Constant-curvature rotation

For a bending plane parameterised by \(\phi\), the bending axis is
\[
\boldsymbol u(\phi)=\begin{bmatrix}-\sin\phi\\ \cos\phi\\ 0\end{bmatrix}=R_z(\phi)\,e_y.
\]
The rotation by bend angle \(\theta\) about \(\boldsymbol u(\phi)\) is realised as
\[
\boxed{R(\phi,\theta)=R_z(\phi)\,R_y(\theta)\,R_z(-\phi)},
\]
which is exactly what the code builds (`_rotz`, `_roty`).

### 1.2 Arc end-point translation in a numerically stable \(s\)–\(\theta\) form

Define the stable functions
\[
A(\theta)=\frac{1-\cos\theta}{\theta},\qquad B(\theta)=\frac{\sin\theta}{\theta}.
\]
For an arc of *active length* \(s\) and bend \(\theta\) in its local bending frame,
\[
\boldsymbol v(\theta,s)=\begin{bmatrix}s\,A(\theta)\\[2pt]0\\[2pt]s\,B(\theta)\end{bmatrix}.
\]
World translation of the arc end-point is \(R_z(\phi)\boldsymbol v(\theta,s)\). The implementation uses robust **series** when \(|\theta|\ll 1\):
\[
\begin{aligned}
A(\theta)&=\tfrac12\theta-\tfrac1{24}\theta^3+\mathcal O(\theta^5), &
A'(\theta)&=\tfrac12-\tfrac18\theta^2+\mathcal O(\theta^4),\\
B(\theta)&=1-\tfrac16\theta^2+\tfrac1{120}\theta^4+\mathcal O(\theta^6), &
B'(\theta)&=-\tfrac13\theta+\tfrac1{30}\theta^3+\mathcal O(\theta^5).
\end{aligned}
\]
(Functions: `_A_B_and_derivs_scalar` / `_A_B_and_derivs_vec`).

### 1.3 Canonical angles

Angles are canonicalised as in the code (`_canon_theta_phi`):
\[
\boxed{\ \theta<0\ \Rightarrow\ (\theta,\phi)\leftarrow(-\theta,\ \phi+\pi)\ },\qquad
\phi\leftarrow \phi \bmod 2\pi.
\]

### 1.4 Circular interval projection and box snapping

- For **angles** with explicit bounds \([\phi_{\min},\phi_{\max}]\) on the circle, the code projects to the **nearest point on the permitted arc** (`_project_angle_to_interval`). If no bounds are given, angles are merely wrapped to \([0,2\pi)\).
- For **scalars** with linear bounds, `_snap_to_box(x,lo,hi,tol)` clamps to \([lo,hi]\) with a small tolerance; values outside by more than `tol` are rejected (return `None`).

---

## 2) Chain ordering and forward kinematics

The solver fixes the outer segment’s **active** length \(s_1\) and passive \(L_{1p}\) as constants for the IK (see §6). For each segment let
\[
R_i=R(\phi_i,\theta_i).
\]

- **Outer**: 
\[
p_1=L_{1p}e_z+R_z(\phi_1)\,\boldsymbol v(\theta_1,s_1),\qquad R_1=R(\phi_1,\theta_1).
\]

- **Inner** (passive → active arc → rigid tip of length \(L_{\text{rigid}}\)):
\[
q=L_{2p}e_z+R_z(\phi_2)\,\boldsymbol v(\theta_2,s_2)+R_2\,L_{\text{rigid}}e_z,\qquad R_2=R(\phi_2,\theta_2).
\]

- **Tip**: 
\[
p_{\text{tip}}=p_1+R_1\,q,\qquad R_{\text{tip}}=R_1R_2,\qquad \boldsymbol b_{\text{world}}=R_{\text{tip}}\boldsymbol b_0.
\]

- **Base feed** \(d\) is applied **along \(+\hat{\boldsymbol z}\)** to the tip position only: \(p_{\text{tip}}^{(d)}=p_{\text{tip}}+d\,e_z\).

---

## 3) Closed-form inner orientation \((\phi_2,\theta_2)\)

The inner segment orientation is solved **in closed form** from the bevel alignment:
\[
\hat{\boldsymbol n}'=R_1^\top\,\hat{\boldsymbol n}_\star.
\]
The code computes (function `_InnerClosedForm.angles_from_orientation`):
\[
\boxed{\ \phi_2=\operatorname{atan2}\!\big(n'_y,\ n'_x-\sin\alpha\big)\ },
\]
then introduces
\[
\boldsymbol u=\begin{bmatrix}\sin\alpha\cos\phi_2\\[2pt]-\sin\alpha\sin\phi_2\\[2pt]\cos\alpha\end{bmatrix},\quad
\boldsymbol w=\begin{bmatrix}\cos\phi_2\,n'_x+\sin\phi_2\,n'_y\\[2pt]-\sin\phi_2\,n'_x+\cos\phi_2\,n'_y\\[2pt]n'_z\end{bmatrix},
\]
and robustly evaluates
\[
\cos\theta_2=\frac{u_x w_x+u_z w_z}{u_x^2+u_z^2},\qquad
\sin\theta_2=\frac{u_z w_x-u_x w_z}{u_x^2+u_z^2},\qquad
\theta_2=\operatorname{atan2}(\sin\theta_2,\cos\theta_2).
\]
If \(|\theta_2|\) is tiny, the implementation returns \(R_2=I\) and \(\hat{\boldsymbol z}_2=e_z\), improving numerical stability.

**Bounds** are applied subsequently: \(\theta_2\) via `_snap_to_box`, \(\phi_2\) projected to its circular interval if present (otherwise wrapped). If \(\phi_2\) must be moved to meet bounds, the candidate may later be rejected by the position/bevel tests; no additional re-solve is performed.

---

## 4) Inner lengths with **fixed** active length \(s_2\) and rigid tip

Unlike earlier formulations, the implementation holds \(s_2\) **fixed** (`s2_fixed`). Only the **pre-bend passive** length \(L_{2p}\) is solved in closed form. Work in the outer-end frame:
\[
\boldsymbol q = R_1^\top\big(P_\star - p_1\big) - R_2\,L_{\text{rigid}}e_z.
\]
With \(A_2=A(\theta_2),\ B_2=B(\theta_2)\) we have
\[
\boxed{\ L_{2p} = q_z - s_2\,B_2 \ } \quad\text{(function `L2p_from_position_fixed_s2`).}
\]
`L_{2p}` is then **snapped** to `[passive_L_min, passive_L_max]` with tolerance; if total inner length would exceed `L_max`, it is reduced to satisfy the cap. This is the **only** positional solve for inner lengths; the \(x\!y\) residuals are handled upstream by the search/refinement over \((\theta_1,\phi_1)\).

---

## 5) Choosing the base feed \(d\)

Given the pose without feed, \(p_{\text{tip}}\), the best feed is the **clamped** shift along \(e_z\):
\[
d^\star = \operatorname{clip}\big(P_{\star,z}-p_{\text{tip},z},\ d_{\min},\,d_{\max}\big).
\]
(Implementation: `_best_d_along_z`.)

---

## 6) Segment length handling (outer constants)

Let `seg1` be the **outer** segment used in IK. During a solve the code treats:
- \(s_1\) as **constant**. If both total and passive lengths are fixed (`L_min=L_max` and `passive_L_min=passive_L_max`), then \(s_1=L_{\text{total}}-L_{\text{passive}}\). Otherwise a midpoint heuristic is used:
\[
s_1=\tfrac12\big((L_{\min}+L_{\max})-(L_{p,\min}+L_{p,\max})\big).
\]
- \(L_{1p}\) as fixed to either the unique passive length, or its midpoint:
\[
L_{1p}=\begin{cases}
L_{p,\max}, & \text{if } L_{p,\min}=L_{p,\max},\\[4pt]
\tfrac12(L_{p,\min}+L_{p,\max}), & \text{otherwise.}
\end{cases}
\]

For the **inner** segment, \(s_2\) is the provided constant `s2_fixed\(\ge 0\)` (validated against the active capacity), and \(L_{2p}\) is solved as in §4 subject to bounds and total length cap.

---

## 7) Acceptance tests and diagnostics

For each candidate, the solver computes:
- **Position error** \(\mathrm{pos\_err}=\|p_{\text{tip}}^{(d)}-P_\star\|\). Must be \(\le \varepsilon_{\text{pos}}\) (`opts.pos_tol`).  
- **Bevel alignment**: \(c_b=\boldsymbol b_{\text{world}}\!\cdot\!\hat{\boldsymbol n}_\star\). Must satisfy \(c_b\ge\cos(\varepsilon_{\text{bevel}})\) where \(\varepsilon_{\text{bevel}}\) is `opts.bevel_tol_deg` in degrees.

If either test fails, the candidate is **discarded**.

A diagnostic axis error (not a hard constraint) is reported:
\[
\mathrm{ang\_err\_deg}=\big|\arccos(\hat{\boldsymbol z}_{\text{inner}}\!\cdot\!\hat{\boldsymbol n}_\star)\,[^\circ]-\mathrm{angle\_target\_deg}\big|,
\]
with `angle_target_deg` default \(45^\circ\). This value is used for tie-breaking (§10).

All angles returned in solutions are **canonicalised** (§1.3).

---

## 8) Search strategy over \((\theta_1,\phi_1)\)

Only two free variables are searched: \(\theta_1\) and \(\phi_1\) (outer bend and plane). Everything else is closed-form or direct (\(\phi_2,\theta_2,L_{2p},d\)) given \((\theta_1,\phi_1)\).

### 8.1 Theta seeding
- Build a coarse grid on \([\theta_{1,\min},\theta_{1,\max}]\) (default 41 samples).
- For each \(\theta_1\), sweep \(\phi_1\) on a uniform ring (48 samples) and score
\[
J=\mathrm{pos\_err}+10^{-3}\,\mathrm{ang\_err\_deg}.
\]
- Keep `keep_top` best thetas (default 6), then refine each by a **local window** \(\pm 5^\circ\) (19 samples). Always include near-straight \(\theta_1=\pm10^{-3}\).

### 8.2 Phi scan and Brent refinement
For each \(\theta_1\) seed:
- Evaluate a dense ring for \(\phi_1\) (72 samples; 144 if \(\theta_1\) is near its bounds). Keep up to 6 best.
- For up to 5 of them, run a bounded **Brent** 1-D search on \(\phi_1\) in \(\pm 6^\circ\) about the seed, minimising
\[
J=\mathrm{pos\_err}+10^{-6}\,\mathrm{ang\_err\_deg}.
\]

### 8.3 Two-variable LM polish
Run a damped **Levenberg–Marquardt** update in \((\theta_1,\phi_1)\) only, using residuals
\[
\boldsymbol r_p=P_\star-p_{\text{tip}}^{(d)},\qquad
\boldsymbol r_b=\hat{\boldsymbol n}_\star-\widehat{\boldsymbol b}_{\text{world}},
\]
stacked as \([\boldsymbol r_p;\sqrt{w_b}\,\boldsymbol r_b]\) with \(w_b=20\). Finite-difference Jacobians are used; parameter updates are **projected to bounds** each step. The scalar objective guiding acceptance is
\[
f=\mathrm{pos\_err}+10^{-3}\,\mathrm{ang\_err\_deg}+10^{-6}\,|d|.
\]
A trust-region style damping schedule halves/doubles the LM \(\lambda\) upon success/failure.

Early termination occurs when a refined candidate already meets **tight** thresholds (5% of position tolerance and \(0.5\,\varepsilon_{\text{bevel}}\)).

---

## 9) Handling of bounds and angles (as in code)

- \(\theta\) bounds: `_snap_to_box` with a tiny angular tolerance. If outside by more than tolerance → candidate rejected.
- \(\phi\) bounds: if both \(\phi_{\min},\phi_{\max}\) are specified, `_project_angle_to_interval` returns the **nearest** feasible value on the circular arc; else angles are simply wrapped to \([0,2\pi)\).
- Passive and total length bounds are enforced for `L_{2p}` and the inner total; if the inner total would exceed `L_max`, `L_{2p}` is reduced accordingly.

---

## 10) Selection and output

All accepted candidates are filtered by a **Pareto front** on 
\[
(\mathrm{pos\_err},\ \mathrm{ang\_err\_deg},\ |d|),
\]
then sorted lexicographically by \(\mathrm{pos\_err}\) then \(\mathrm{ang\_err\_deg}\). The top-`k` (`opts.topk`) are returned as `IKSolution` objects carrying:
- `reachable=True`,
- errors, translation \(d\),
- per-segment `SegmentSolution` (canonical angles; fixed outer lengths; inner `L_{2p}` solved, `s_2` fixed),
- homogeneous tip transform `end_T` with rotation \(R_1R_2\) and **position without feed** \(p_{\text{tip}}\) (the feed \(d\) is reported separately).

The `meta` payload includes: world bevel vector, inner axis direction, rigid tip, end position with feed, and a human-readable report of bend/rotation/translation per segment.

---

## 11) Analytical Jacobians (for reference and future extensions)

Although the current LM uses finite differences in \((\theta_1,\phi_1)\), the codebase provides stable building blocks for analytical derivatives:

### 11.1 Rotation derivatives
For \(R(\phi,\theta)=R_z(\phi)R_y(\theta)R_z(-\phi)\),
\[
\frac{\partial R}{\partial \phi}=\widehat{e_z}\,R-R\,\widehat{e_z},\qquad
\frac{\partial R}{\partial \theta}=R\,\widehat{R_z(\phi)e_y}.
\]

### 11.2 Arc translation derivatives
\[
\frac{\partial\boldsymbol v}{\partial s}=\begin{bmatrix}A(\theta)\\0\\B(\theta)\end{bmatrix},\qquad
\frac{\partial\boldsymbol v}{\partial\theta}=s\begin{bmatrix}A'(\theta)\\0\\B'(\theta)\end{bmatrix},
\]
with series forms near \(\theta=0\).

### 11.3 Position Jacobian structure
With \(p=p_1+R_1 q + d\,e_z\), one obtains (matching the implementation’s building blocks):
\[
\begin{aligned}
\frac{\partial p}{\partial \theta_1} &= R_z(\phi_1)\,\partial_\theta\boldsymbol v(\theta_1,s_1)+(\partial_{\theta_1}R_1)\,q,\\
\frac{\partial p}{\partial \phi_1}   &= R_z(\phi_1)\,\widehat{e_z}\,\boldsymbol v(\theta_1,s_1)+(\partial_{\phi_1}R_1)\,q,\\
\frac{\partial p}{\partial \theta_2} &= R_1\!\left(R_z(\phi_2)\,\partial_\theta\boldsymbol v(\theta_2,s_2)+(\partial_{\theta_2}R_2)\,L_{\text{rigid}}e_z\right),\\
\frac{\partial p}{\partial \phi_2}   &= R_1\!\left(R_z(\phi_2)\,\widehat{e_z}\,\boldsymbol v(\theta_2,s_2)+(\partial_{\phi_2}R_2)\,L_{\text{rigid}}e_z\right),\\
\frac{\partial p}{\partial s_2}      &= R_1\!\left(R_z(\phi_2)\,\partial_s\boldsymbol v(\theta_2,s_2)\right),\\
\frac{\partial p}{\partial L_{2p}}   &= R_1 e_z,\qquad
\frac{\partial p}{\partial d}=e_z.
\end{aligned}
\]

### 11.4 Bevel Jacobian structure
\[
\boldsymbol b=R_1R_2\boldsymbol b_0,\quad
\frac{\partial \boldsymbol b}{\partial \theta_1}=(\partial_{\theta_1}R_1)\,R_2\boldsymbol b_0,\ 
\frac{\partial \boldsymbol b}{\partial \phi_1}=(\partial_{\phi_1}R_1)\,R_2\boldsymbol b_0,\ 
\frac{\partial \boldsymbol b}{\partial \theta_2}=R_1(\partial_{\theta_2}R_2)\boldsymbol b_0,\ 
\frac{\partial \boldsymbol b}{\partial \phi_2}=R_1(\partial_{\phi_2}R_2)\boldsymbol b_0.
\]
Project to the tangent space via \(I-\boldsymbol b\boldsymbol b^\top\) if needed.

---

## 12) Numerical and implementation notes

- All vector normals are **re-normalised** using a small-norm guard (`_normalize`).
- Small denominators in the inner orientation solve use a positive floor \(10^{-16}\).
- The bounded Brent implementation is simple and robust for the 1-D \(\phi_1\) refine.
- The code respects provided angular/length bounds exactly as described; **rejected** candidates are not further repaired.
- All outputs use canonical angles (§1.3) before packaging `SegmentSolution` objects.

---
