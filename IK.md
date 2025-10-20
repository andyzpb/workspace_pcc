# IK pipeline documentation for two-segment PCC robot

## 0) Problem

Given a target contact point \(P_\star\in\mathbb{R}^3\) and a target surface normal \(\hat{\boldsymbol n}_\star\in\mathbb{S}^2\), find a feasible configuration of a two-segment PCC robot (**outer**, **inner**) with **no roll actuation** and a **base feed** \(d\) along world \(+\hat{\boldsymbol z}\) (which affects position only), such that
\[
\|\,p_{\text{tip}}^{(d)}-P_\star\,\|\le \varepsilon_{\text{pos}},\qquad
\angle(\boldsymbol b_{\text{world}},\,\hat{\boldsymbol n}_\star)\le \varepsilon_{\text{bevel}},
\]
where \(p_{\text{tip}}^{(d)} = p_{\text{tip}} + d\,\hat{\boldsymbol z}\).
In implementation, we equivalently move \(d\) to the target side and solve against
\(P_\star^{(d)} := P_\star - d\,\hat{\boldsymbol z}\).

The bevel direction is \(\boldsymbol b_{\text{world}}=R_{\text{tip}}\boldsymbol b_0\) with
\[
\boldsymbol b_0=\begin{bmatrix}\sin\alpha\\[2pt]0\\[2pt]\cos\alpha\end{bmatrix},
\]
and \(\alpha\) the known bevel angle (field `bevel_angle_deg`).

---

## 1) PCC Kinematics

### 1.1 Constant-curvature rotation

Let the bending axis be
\[
\boldsymbol u(\phi)=[-\sin\phi,\ \cos\phi,\ 0]^\top=R_z(\phi)e_y.
\]
A rotation by angle \(\theta\) about \(\boldsymbol u(\phi)\) is
\[
R(\phi,\theta)=\exp\!\big(\theta\,\widehat{\boldsymbol u(\phi)}\big)
=R_z(\phi)\exp(\theta\widehat{e_y})R_z(-\phi)
=\boxed{\,R_z(\phi)\,R_y(\theta)\,R_z(-\phi)\,}.
\]
The code also provides a Rodrigues axis–angle form (`_axis_angle_R`):
\[
R = cI+(1-c)\,\boldsymbol u\boldsymbol u^\top + s\,\widehat{\boldsymbol u},\quad
c=\cos\theta,\ s=\sin\theta.
\]

### 1.2 Arc endpoint translation & straight-line limit

In the local bending plane (axis \(e_y\)), a circular arc of radius \(r\) and angle \(\theta\) has endpoint
\[
\boldsymbol p_0(r,\theta)=
\begin{bmatrix}r(1-\cos\theta)\\[2pt]0\\[2pt]r\sin\theta\end{bmatrix}.
\]
In world frame:
\[
\boxed{\ \boldsymbol p(\phi,\kappa,\theta)=R_z(\phi)\,\boldsymbol p_0\!\big(r=\tfrac1\kappa,\theta\big)\ }.
\]
The implementation `_cc_transform(phi,kappa,theta)` handles \(\kappa\!\approx\!0\) or \(\theta\!\approx\!0\) via a **first-order consistent** straight-line limit:
\[
\boldsymbol p_0 \approx \begin{bmatrix}\tfrac12 s\,\theta\\[2pt]0\\[2pt]s\end{bmatrix},\qquad s=\theta/\kappa.
\]

### 1.3 Stable \(s\)–\(\theta\) form and derivatives

Define
\[
\boldsymbol v(\theta,s)=\begin{bmatrix}s\,A(\theta)\\[2pt]0\\[2pt]s\,B(\theta)\end{bmatrix},\quad
A(\theta)=\frac{1-\cos\theta}{\theta},\ \ B(\theta)=\frac{\sin\theta}{\theta}.
\]
Exact derivatives used in code:
\[
A'(\theta)=\frac{\theta\sin\theta-(1-\cos\theta)}{\theta^2},\qquad
B'(\theta)=\frac{\theta\cos\theta-\sin\theta}{\theta^2}.
\]
For \(|\theta|\ll1\) we use series (numerically robust):
\[
\begin{aligned}
A(\theta)&=\tfrac12\theta-\tfrac1{24}\theta^3+\mathcal O(\theta^5),&
A'(\theta)&=\tfrac12-\tfrac18\theta^2+\mathcal O(\theta^4),\\
B(\theta)&=1-\tfrac16\theta^2+\tfrac1{120}\theta^4+\mathcal O(\theta^6),&
B'(\theta)&=-\tfrac13\theta+\tfrac1{30}\theta^3+\mathcal O(\theta^5).
\end{aligned}
\]

### 1.4 Canonical angle representation (`_canon_theta_phi`)

We enforce a canonical representation
\[
\boxed{\ \theta<0\ \Rightarrow\ (\theta,\phi)\leftarrow(-\theta,\ \phi+\pi)\ },\qquad
\phi\leftarrow\phi\bmod 2\pi,
\]
so bends are non-negative and \(\phi\in[0,2\pi)\).

### 1.5 Circular-interval projection for \(\phi\)

Given a circular interval \([\phi_{\min},\phi_{\max}]\) modulo \(2\pi\), project any \(\phi\) to its **nearest** point on the arc:
let \(S=(\phi_{\max}-\phi_{\min})\bmod2\pi\).
If \(S\ge 2\pi-\epsilon\), any \(\phi\) is accepted (only wrap).
Else set \(x=(\phi-\phi_{\min})\bmod2\pi\). If \(x\le S\), already inside.
Otherwise, compare geodesic distances to the arc endpoints, and snap to the nearer endpoint.

---

## 2) Hardware-faithful chain ordering

- **Outer segment** (passive → active):
\[
T_1=\underbrace{\mathrm{Trans}(0,0,L_{1p})}_{T_{\text{pass,1}}}\,
\underbrace{\begin{bmatrix}R(\phi_1,\theta_1)&R_z(\phi_1)\boldsymbol v(\theta_1,s_1)\\ \boldsymbol 0^\top&1\end{bmatrix}}_{T_{\text{act,1}}},\quad
R_1=R_z(\phi_1)R_y(\theta_1)R_z(-\phi_1).
\]

- **Inner segment** (**pre-bend passive → active arc → rigid tip**):
\[
T_2=\underbrace{\mathrm{Trans}(0,0,L_{2p})}_{T_{\text{pass,2}}}\,
\underbrace{\begin{bmatrix}R(\phi_2,\theta_2)&R_z(\phi_2)\boldsymbol v(\theta_2,s_2)\\ \boldsymbol 0^\top&1\end{bmatrix}}_{T_{\text{act,2}}}\,
\underbrace{\mathrm{Trans}(0,0,L_{\text{rigid}})}_{T_{\text{rigid}}},
\]
with \(R_2=R_z(\phi_2)R_y(\theta_2)R_z(-\phi_2)\).

- **Tip**: \(T_{\text{tip}}=T_1T_2\), \(p_{\text{tip}}=T_{\text{tip}}(1{:}3,4)\), \(R_{\text{tip}}=R_1R_2\).  
- **Base feed**: \(p_{\text{tip}}^{(d)}=p_{\text{tip}}+d\,\hat{\boldsymbol z}\). In code we subtract \(d\) from \(P_\star\).

---

## 3) Closed-form inner orientation \((\phi_2,\theta_2)\)

Express the target normal in the outer-end frame:
\[
\hat{\boldsymbol n}'=R_1^\top \hat{\boldsymbol n}_\star.
\]
We seek \(R_2=R_z(\phi_2)R_y(\theta_2)R_z(-\phi_2)\) such that
\[
R_2\,\boldsymbol b_0 \parallel \hat{\boldsymbol n}'.
\]

### 3.1 Choosing \(\phi_2\) (rotation about \(y\) preserves the \(y\)-component)

With
\[
\boldsymbol u=R_z(-\phi_2)\boldsymbol b_0=\begin{bmatrix}\sin\alpha\cos\phi_2\\[2pt]-\sin\alpha\sin\phi_2\\[2pt]\cos\alpha\end{bmatrix},\ 
\boldsymbol w=R_z(-\phi_2)\hat{\boldsymbol n}'=\begin{bmatrix}c\,n'_x+s\,n'_y\\[2pt]-s\,n'_x+c\,n'_y\\[2pt]n'_z\end{bmatrix},
\]
\(c=\cos\phi_2,\ s=\sin\phi_2\). Equality of \(y\) components gives
\[
-\sin\alpha\sin\phi_2=-\sin\phi_2\,n'_x+\cos\phi_2\,n'_y
\ \Longrightarrow\
\boxed{\ \phi_2=\operatorname{atan2}\!\big(n'_y,\ n'_x-\sin\alpha\big)\ }.
\]

### 3.2 Solving \(\theta_2\) in the \(xz\) plane

In the de-rotated frame, \(R_y(\theta_2)\) acts only in \(xz\):
\[
\cos\theta_2=\frac{u_x w_x+u_z w_z}{u_x^2+u_z^2},\qquad
\sin\theta_2=\frac{u_z w_x-u_x w_z}{u_x^2+u_z^2},
\]
\[
\boxed{\ \theta_2=\operatorname{atan2}\!\big(u_z w_x-u_x w_z,\ u_x w_x+u_z w_z\big)\ }.
\]
Finally \(R_2=R_z(\phi_2)R_y(\theta_2)R_z(-\phi_2)\), \(\hat{\boldsymbol z}_2=R_2 e_z\).
A small positive floor is used on the denominator for numerical robustness.

---

## 4) Closed-form inner lengths \((s_2,L_{2p})\) with rigid tip

Work in the outer-end frame \((P_1,R_1)\). Define
\[
\boldsymbol q=R_1^\top\big(P_\star^{(d)}-P_1\big),\quad
\boldsymbol a=R_z(\phi_2)\!\begin{bmatrix}1-\cos\theta_2\\[2pt]0\\[2pt]\sin\theta_2\end{bmatrix},\quad
\hat{\boldsymbol z}_2=R_2 e_z.
\]
Geometry of **pre-bend passive → active arc → rigid** yields
\[
\boxed{\ \boldsymbol q=\underbrace{\begin{bmatrix}0\\0\\L_{2p}\end{bmatrix}}_{\text{pre-bend passive}}+\underbrace{r\,\boldsymbol a}_{\text{active arc}}+\underbrace{L_{\text{rigid}}\hat{\boldsymbol z}_2}_{\text{rigid tip}}\ },\qquad r=\frac{s_2}{\theta_2}.
\]
Subtract the rigid contribution: \(\boldsymbol q_2=\boldsymbol q-L_{\text{rigid}}\hat{\boldsymbol z}_2\). The \(xy\) components satisfy \(\boldsymbol q_{2,xy}=r\,\boldsymbol a_{xy}\). Least-squares projection gives
\[
\boxed{\ r=\frac{\boldsymbol a_{xy}^\top\boldsymbol q_{2,xy}}{\|\boldsymbol a_{xy}\|^2}}\quad(\|\boldsymbol a_{xy}\|>\tau).
\]
Then
\[
\boxed{\ s_2=\theta_2\,r,\qquad L_{2p}=q_{2,z}-r\,a_z. }
\]
Degenerate case: if \(\|\boldsymbol a_{xy}\|\) or \(|\theta_2|\) is tiny, take \(r=0\Rightarrow s_2=0\), \(L_{2p}=q_{2,z}\). Afterwards project to bounds (§8).

---

## 5) Forward kinematics in compact, stable form

Using \(\boldsymbol v\) of §1.3,
\[
\begin{aligned}
p_1 &= L_{1p}e_z + R_z(\phi_1)\,\boldsymbol v(\theta_1,s_1),\\
q   &= L_{2p}e_z + R_z(\phi_2)\,\boldsymbol v(\theta_2,s_2) + R_2\,L_{\text{rigid}}e_z,\\
p   &= p_{\text{tip}} = p_1 + R_1 q,\qquad R_{\text{tip}}=R_1R_2,\qquad \boldsymbol b_{\text{world}}=R_{\text{tip}}\boldsymbol b_0,\\
p_{\text{tip}}^{(d)}&=p + d\,e_z.
\end{aligned}
\]

---

## 6) Analytical Jacobians — full derivations

We differentiate with respect to
\[
\boldsymbol x=\big[\theta_1,\phi_1,\theta_2,\phi_2,s_2,L_{2p},d\big]^\top.
\]

### 6.1 Rotation derivatives

For \(R(\phi,\theta)=R_z(\phi)R_y(\theta)R_z(-\phi)\), with
\(\tfrac{\mathrm d}{\mathrm d\phi}R_z(\phi)=R_z(\phi)\widehat{e_z}\) and
\(\tfrac{\mathrm d}{\mathrm d\theta}R_y(\theta)=R_y(\theta)\widehat{e_y}\),
and using product rule,
\[
\begin{aligned}
\frac{\partial R}{\partial \phi}
&=\widehat{e_z}\,R - R\,\widehat{e_z},\\
\frac{\partial R}{\partial \theta}
&=R\,\big(R_z(\phi)\widehat{e_y}R_z(-\phi)\big)=\boxed{R\,\widehat{R_z(\phi)e_y}}.
\end{aligned}
\]
These identities apply to both \(R_1(\phi_1,\theta_1)\) and \(R_2(\phi_2,\theta_2)\).

### 6.2 Arc-translation derivatives

From §1.3,
\[
\frac{\partial\boldsymbol v}{\partial s}=\begin{bmatrix}A(\theta)\\[2pt]0\\[2pt]B(\theta)\end{bmatrix},\qquad
\frac{\partial\boldsymbol v}{\partial \theta}=s\begin{bmatrix}A'(\theta)\\[2pt]0\\[2pt]B'(\theta)\end{bmatrix},
\]
with series substitutions near \(\theta=0\) for stability.

### 6.3 Position Jacobian \(J_p=\partial p/\partial \boldsymbol x\)

Recall \(p=p_1+R_1 q + d\,e_z\), where
\(p_1=L_{1p}e_z+R_z(\phi_1)\boldsymbol v(\theta_1,s_1)\) (here \(s_1,L_{1p}\) are treated as fixed within the solver) and \(q=L_{2p}e_z+R_z(\phi_2)\boldsymbol v(\theta_2,s_2)+R_2L_{\text{rigid}}e_z\).
Then
\[
\begin{aligned}
\frac{\partial p}{\partial \theta_1} &= R_z(\phi_1)\frac{\partial\boldsymbol v(\theta_1,s_1)}{\partial \theta} + \frac{\partial R_1}{\partial \theta_1}\,q,\\
\frac{\partial p}{\partial \phi_1}   &= R_z(\phi_1)\widehat{e_z}\,\boldsymbol v(\theta_1,s_1) + \frac{\partial R_1}{\partial \phi_1}\,q,\\
\frac{\partial p}{\partial \theta_2} &= R_1\!\left(R_z(\phi_2)\frac{\partial\boldsymbol v(\theta_2,s_2)}{\partial \theta} + \frac{\partial R_2}{\partial \theta_2}L_{\text{rigid}}e_z\right),\\
\frac{\partial p}{\partial \phi_2}   &= R_1\!\left(R_z(\phi_2)\widehat{e_z}\,\boldsymbol v(\theta_2,s_2) + \frac{\partial R_2}{\partial \phi_2}L_{\text{rigid}}e_z\right),\\
\frac{\partial p}{\partial s_2}      &= R_1\!\left(R_z(\phi_2)\frac{\partial\boldsymbol v(\theta_2,s_2)}{\partial s}\right),\\
\frac{\partial p}{\partial L_{2p}}   &= R_1 e_z,
\\\frac{\partial p}{\partial d} &=e_z.
\end{aligned}
\]

### 6.4 Orientation (bevel) Jacobian \(J_b=\partial \boldsymbol b/\partial \boldsymbol x\)

With \(\boldsymbol b=R_1R_2\boldsymbol b_0\) (unit by construction),
\[
\frac{\partial \boldsymbol b}{\partial \theta_1}=(\partial R_1/\partial\theta_1)\,R_2\boldsymbol b_0,\quad
\frac{\partial \boldsymbol b}{\partial \phi_1}  =(\partial R_1/\partial\phi_1)\,R_2\boldsymbol b_0,\quad
\frac{\partial \boldsymbol b}{\partial \theta_2}=R_1\,(\partial R_2/\partial\theta_2)\boldsymbol b_0,\quad
\frac{\partial \boldsymbol b}{\partial \phi_2}  =R_1\,(\partial R_2/\partial\phi_2)\boldsymbol b_0.
\]
Derivatives w.r.t. \(s_2,L_{2p},d\) are \(\boldsymbol 0\). If the code normalises \(\boldsymbol b\), optionally project
\[
J_b\leftarrow(I-\boldsymbol b\boldsymbol b^\top)J_b
\]
to the tangent space of \(\mathbb{S}^2\).

---

## 7) Gauss–Newton / QP refinement (as implemented)

Residuals:
\[
\boldsymbol r_p(\boldsymbol x)=p(\boldsymbol x)+d\,e_z-P_\star,\qquad
\boldsymbol r_b(\boldsymbol x)=\boldsymbol b(\boldsymbol x)-\hat{\boldsymbol n}_\star.
\]
Objective (with bevel weight \(w_b\) and Tikhonov \(\lambda\)):
\[
J(\boldsymbol x)=\|\boldsymbol r_p\|^2 + w_b\|\boldsymbol r_b\|^2 + \lambda\|\Delta\boldsymbol x\|^2.
\]
Normal equations:
\[
A\Delta=g,\quad A=J_p^\top J_p + w_b J_b^\top J_b + \lambda I,\quad g=J_p^\top \boldsymbol r_p + w_b J_b^\top \boldsymbol r_b.
\]
We cap \(\|\Delta\|\) (trust region via backtracking Armijo line search), **project** updated parameters to bounds/circular intervals, and **rebuild \(T_1\)** before re-evaluation (this is essential as the inner closed forms depend on \(R_1\)).

---