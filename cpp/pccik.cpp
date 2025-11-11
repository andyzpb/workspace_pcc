#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#if defined(__APPLE__)
#include <math.h>
#include <pthread.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

#ifndef PCC_REAL
#define PCC_REAL double
#endif
using Real = PCC_REAL;

static inline void set_thread_qos_interactive() noexcept {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
#define PCC_ALWAYS_INLINE inline __attribute__((always_inline))
#define PCC_HOT __attribute__((hot))
#define PCC_COLD __attribute__((cold))
#define PCC_FLATTEN __attribute__((flatten))
#else
#define PCC_ALWAYS_INLINE inline
#define PCC_HOT
#define PCC_COLD
#define PCC_FLATTEN
#endif

namespace detail {
template <class T> constexpr T sqr(T x) noexcept { return x * x; }
template <class T> inline constexpr T two_pi_v = T(2) * std::numbers::pi_v<T>;
template <class T> inline constexpr T eps_v = std::numeric_limits<T>::epsilon();
template <class T> PCC_ALWAYS_INLINE T fmadd(const T&a,const T&b,const T&c){
  if constexpr (std::is_arithmetic_v<T>) return std::fma(a,b,c);
  else return a*b + c;
}
template <class T>
PCC_ALWAYS_INLINE void fast_sincos(T a, T &s, T &c) noexcept {
#if (defined(__GNUC__) || defined(__clang__))
#if defined(__APPLE__)
  if constexpr (std::is_same_v<T, double>) { ::__sincos(a, &s, &c); return; }
  if constexpr (std::is_same_v<T, float>)  { ::__sincosf(a, &s, &c); return; }
#else
  if constexpr (std::is_same_v<T, double>) { ::sincos(a, &s, &c); return; }
  if constexpr (std::is_same_v<T, float>)  { ::sincosf(a, &s, &c); return; }
#endif
#endif
  using std::sin; using std::cos;
  s = sin(a); c = cos(a);
}
template <class T> PCC_ALWAYS_INLINE T wrap_0_2pi(T a) noexcept {
  const T twopi = two_pi_v<T>, inv = T(1) / twopi;
  T t = a - std::floor(a * inv) * twopi;
  if (t < T(0)) t += twopi;
  if (t >= twopi) t -= twopi;
  return t;
}
template <class T>
PCC_ALWAYS_INLINE T project_angle_to_interval(T phi, T lo, T hi) noexcept {
  lo = wrap_0_2pi(lo); hi = wrap_0_2pi(hi);
  const T twopi = two_pi_v<T>;
  const T span = std::fmod(hi - lo + twopi, twopi);
  if (span >= twopi - T(1e-12)) return wrap_0_2pi(phi);
  T x = std::fmod(wrap_0_2pi(phi) - lo + twopi, twopi);
  if (x <= span) return lo + x;
  const T to_lo = std::fmod(twopi - x, twopi);
  const T to_hi = std::fmod(x - span, twopi);
  return (to_hi <= to_lo) ? hi : lo;
}
template <class T>
PCC_ALWAYS_INLINE bool snap_to_box(T &x, T lo, T hi, T tol) noexcept {
  if (x < lo - tol || x > hi + tol) return false;
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return true;
}
template <class T>
PCC_ALWAYS_INLINE void A_B_scalar_stable_ab(T t, T &A, T &B) noexcept {
  const T at = std::abs(t);
  const T small = T(8) * std::cbrt(eps_v<T>);
  if (at < small) {
    const T t2=t*t, t3=t2*t, t4=t2*t2, t5=t4*t, t6=t3*t3, t7=t6*t, t8=t4*t4;
    A = std::fma(T(-1.0/40320.0),t7,std::fma(T(1.0/720.0),t5,std::fma(T(-1.0/24.0),t3,T(0.5)*t)));
    B = std::fma(T(1.0/362880.0),t8,std::fma(T(-1.0/5040.0),t6,std::fma(T(1.0/120.0),t4,std::fma(T(-1.0/6.0),t2,T(1)))));
    return;
  }
  T s,c; fast_sincos(T(0.5)*t, s, c);
  const T s2 = s*s;
  A = (T(2)*s2)/t;
  B = (T(2)*s*c)/t;
}
template <class T>
PCC_ALWAYS_INLINE void A_B_scalar_stable_full(T t, T &A, T &B, T &Ap, T &Bp) noexcept {
  const T at = std::abs(t);
  const T small = T(8) * std::cbrt(eps_v<T>);
  if (at < small) {
    const T t2=t*t, t3=t2*t, t4=t2*t2, t5=t4*t, t6=t3*t3, t7=t6*t, t8=t4*t4;
    A  = std::fma(T(-1.0/40320.0),t7,std::fma(T(1.0/720.0),t5,std::fma(T(-1.0/24.0),t3,T(0.5)*t)));
    B  = std::fma(T(1.0/362880.0),t8,std::fma(T(-1.0/5040.0),t6,std::fma(T(1.0/120.0),t4,std::fma(T(-1.0/6.0),t2,T(1)))));
    Ap = std::fma(T(-1.0/5760.0), t6,std::fma(T(1.0/144.0), t4,std::fma(T(-1.0/8.0), t2,T(0.5))));
    Bp = std::fma(T(1.0/45360.0), t7,std::fma(T(-1.0/840.0), t5,std::fma(T(1.0/30.0), t3,T(-1.0/3.0)*t)));
    return;
  }
  T s,c; fast_sincos(T(0.5)*t, s, c);
  const T s2 = s*s, t2=t*t, c2_minus_s2 = c*c - s2;
  A  = (T(2)*s2)/t;
  B  = (T(2)*s*c)/t;
  Ap = (T(2)*s*c*t - T(2)*s2) / t2;
  Bp = (t * c2_minus_s2 - T(2)*s*c) / t2;
}
}

template<class T> struct Dual2{
  T v, d0, d1;
  Dual2()=default;
  Dual2(T v_):v(v_),d0(0),d1(0){}
  Dual2(T v_,T a,T b):v(v_),d0(a),d1(b){}
};
template<class T> PCC_ALWAYS_INLINE Dual2<T> operator+(const Dual2<T>&a,const Dual2<T>&b){return {a.v+b.v,a.d0+b.d0,a.d1+b.d1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> operator-(const Dual2<T>&a,const Dual2<T>&b){return {a.v-b.v,a.d0-b.d0,a.d1-b.d1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> operator*(const Dual2<T>&a,const Dual2<T>&b){return {a.v*b.v,a.d0*b.v+a.v*b.d0,a.d1*b.v+a.v*b.d1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> operator/(const Dual2<T>&a,const Dual2<T>&b){T inv=1/b.v;T v=a.v*inv;T t=inv*inv;return {v,(a.d0*b.v-a.v*b.d0)*t,(a.d1*b.v-a.v*b.d1)*t};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> operator-(const Dual2<T>&a){return {T(-1)*a.v,T(-1)*a.d0,T(-1)*a.d1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> fma(const Dual2<T>&x,const Dual2<T>&y,const Dual2<T>&z){return x*y+z;}
template<class T> PCC_ALWAYS_INLINE Dual2<T> sin(const Dual2<T>&a){T sv=std::sin(a.v), cv=std::cos(a.v); return {sv,cv*a.d0,cv*a.d1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> cos(const Dual2<T>&a){T cv=std::cos(a.v), sv=std::sin(a.v); return {cv,-sv*a.d0,-sv*a.d1};}
template<class T> PCC_ALWAYS_INLINE void fast_sincos(const Dual2<T>&a, Dual2<T>&s, Dual2<T>&c){s=sin(a); c=cos(a);}
template<class T> PCC_ALWAYS_INLINE Dual2<T> sqrt(const Dual2<T>&a){T r=std::sqrt(a.v); T inv= (r>0)? (T(0.5)/r):T(0); return {r, a.d0*inv, a.d1*inv};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> atan2(const Dual2<T>&y,const Dual2<T>&x){T den = x.v*x.v + y.v*y.v; T v=std::atan2(y.v,x.v); T g0=(x.v*y.d0 - y.v*x.d0)/den; T g1=(x.v*y.d1 - y.v*x.d1)/den; return {v,g0,g1};}
template<class T> PCC_ALWAYS_INLINE Dual2<T> hypot(const Dual2<T>&x,const Dual2<T>&y){return sqrt(Dual2<T>(x.v*x.v + y.v*y.v, 2*x.v*x.d0 + 2*y.v*y.d0, 2*x.v*x.d1 + 2*y.v*y.d1));}
template<class T> PCC_ALWAYS_INLINE Dual2<T> abs(const Dual2<T>&a){return {std::abs(a.v), (a.v>=0? a.d0 : -a.d0), (a.v>=0? a.d1 : -a.d1)};}
template<class S> PCC_ALWAYS_INLINE Real valof(const S& x){ if constexpr (std::is_arithmetic_v<S>) return static_cast<Real>(x); else return x.v; }
template<class S> PCC_ALWAYS_INLINE S wrap0_any(const S& a){ if constexpr (std::is_arithmetic_v<S>) return detail::wrap_0_2pi(a); else return S(detail::wrap_0_2pi(a.v), a.d0, a.d1); }
template<class U> PCC_ALWAYS_INLINE void fsincos(const U& a, U& s, U& c){ if constexpr (std::is_arithmetic_v<U>) detail::fast_sincos(a,s,c); else ::fast_sincos(a,s,c); }
template<class T> PCC_ALWAYS_INLINE T clamp_scalar(T x, T lo, T hi){ if (x<lo) x=lo; if (x>hi) x=hi; return x; }
template<class S> PCC_ALWAYS_INLINE S clamp_any(const S& x, Real lo, Real hi){ if constexpr (std::is_arithmetic_v<S>) return clamp_scalar<S>(x, S(lo), S(hi)); else return (x.v<=lo)? S(lo,0,0) : (x.v>=hi)? S(hi,0,0) : x; }

template <class T> struct Vec3{
  T x,y,z;
  PCC_ALWAYS_INLINE Vec3 operator+(const Vec3&o) const noexcept { return {x+o.x,y+o.y,z+o.z}; }
  PCC_ALWAYS_INLINE Vec3 operator-(const Vec3&o) const noexcept { return {x-o.x,y-o.y,z-o.z}; }
  PCC_ALWAYS_INLINE Vec3 operator*(T s) const noexcept { return {x*s,y*s,z*s}; }
  PCC_ALWAYS_INLINE T dot(const Vec3&o) const noexcept { return detail::fmadd(x,o.x, detail::fmadd(y,o.y, z*o.z)); }
  PCC_ALWAYS_INLINE T norm2() const noexcept { return dot(*this); }
  PCC_ALWAYS_INLINE T norm() const noexcept { using std::sqrt; return sqrt(norm2()); }
  PCC_ALWAYS_INLINE Vec3 normalized(T eps=T(1e-12)) const noexcept {
    const T n=norm(); if(valof(n)>valof(eps)){T inv=T(1)/n;return {x*inv,y*inv,z*inv};}
    return {T(0),T(0),T(0)};
  }
};

template <class T> struct Mat3{
  T a00,a01,a02,a10,a11,a12,a20,a21,a22;
  PCC_ALWAYS_INLINE Vec3<T> apply(const Vec3<T>&v) const noexcept {
    return { detail::fmadd(a02,v.z,detail::fmadd(a01,v.y,a00*v.x)),
             detail::fmadd(a12,v.z,detail::fmadd(a11,v.y,a10*v.x)),
             detail::fmadd(a22,v.z,detail::fmadd(a21,v.y,a20*v.x)) };
  }
  PCC_ALWAYS_INLINE Mat3 mul(const Mat3&B) const noexcept {
    const Mat3&A=*this;
    return {
      detail::fmadd(A.a01,B.a10,detail::fmadd(A.a02,B.a20,A.a00*B.a00)),
      detail::fmadd(A.a01,B.a11,detail::fmadd(A.a02,B.a21,A.a00*B.a01)),
      detail::fmadd(A.a01,B.a12,detail::fmadd(A.a02,B.a22,A.a00*B.a02)),
      detail::fmadd(A.a11,B.a10,detail::fmadd(A.a12,B.a20,A.a10*B.a00)),
      detail::fmadd(A.a11,B.a11,detail::fmadd(A.a12,B.a21,A.a10*B.a01)),
      detail::fmadd(A.a11,B.a12,detail::fmadd(A.a12,B.a22,A.a10*B.a02)),
      detail::fmadd(A.a21,B.a10,detail::fmadd(A.a22,B.a20,A.a20*B.a00)),
      detail::fmadd(A.a21,B.a11,detail::fmadd(A.a22,B.a21,A.a20*B.a01)),
      detail::fmadd(A.a21,B.a12,detail::fmadd(A.a22,B.a22,A.a20*B.a02))
    };
  }
  static PCC_ALWAYS_INLINE Mat3 rotZ(T a) noexcept { T s,c; fsincos(a,s,c); return {c,-s,0,s,c,0,0,0,1}; }
  static PCC_ALWAYS_INLINE Mat3 rotY(T a) noexcept { T s,c; fsincos(a,s,c); return {c,0,s,0,1,0,-s,0,c}; }
};

template <class T>
PCC_ALWAYS_INLINE Mat3<T> RzRyRz(T phi, T theta) noexcept {
  T sp,cp; fsincos(phi,sp,cp);
  T st,ct; fsincos(theta,st,ct);
  const T r00=cp,r01=-sp,r02=0,r10=sp,r11=cp,r12=0,r20=0,r21=0,r22=1;
  const Mat3<T> Ry={ct,0,st,0,1,0,-st,0,ct};
  const Mat3<T> Rzm={cp,sp,0,-sp,cp,0,0,0,1};
  const Mat3<T> Rzr={
    r00*Ry.a00 + r01*Ry.a10 + r02*Ry.a20,
    r00*Ry.a01 + r01*Ry.a11 + r02*Ry.a21,
    r00*Ry.a02 + r01*Ry.a12 + r02*Ry.a22,
    r10*Ry.a00 + r11*Ry.a10 + r12*Ry.a20,
    r10*Ry.a01 + r11*Ry.a11 + r12*Ry.a21,
    r10*Ry.a02 + r11*Ry.a12 + r12*Ry.a22,
    r20*Ry.a00 + r21*Ry.a10 + r22*Ry.a20,
    r20*Ry.a01 + r21*Ry.a11 + r22*Ry.a21,
    r20*Ry.a02 + r21*Ry.a12 + r22*Ry.a22
  };
  return Rzr.mul(Rzm);
}

struct SegmentBounds{
  Real theta_min{}, theta_max{};
  bool has_phi_bounds{}; Real phi_min{}, phi_max{};
  Real L_min{}, L_max{};
  Real passive_L_min{}, passive_L_max{};
  Real active_L_max{};
  Real bevel_angle_deg{Real(45)};
  Real rigid_tip_length{};
};
struct SolverConst{
  Real s1{}, L1p{};
  Real s2_fixed{};
  Real d_min{}, d_max{};
  Real pos_tol{Real(1e-5)};
  Real bevel_tol_deg{Real(1e-5)};
  Real angle_target_deg{Real(45)};
};

static inline Real wrap0(Real a){ return detail::wrap_0_2pi(a); }
static inline Real proj_ang(Real a, Real lo, Real hi){ return detail::project_angle_to_interval(a,lo,hi); }
static inline bool clamp_box(Real &x, Real lo, Real hi, Real tol){ return detail::snap_to_box(x,lo,hi,tol); }

template<class S>
PCC_ALWAYS_INLINE void A_B_generic(S t, S& A, S& B){
  using std::sin; using std::cos;
  S h = S(0.5)*t;
  auto s = sin(h), c = cos(h);
  A = (S(2)*s*s)/t;
  B = (S(2)*s*c)/t;
}
template<>
PCC_ALWAYS_INLINE void A_B_generic<Real>(Real t, Real& A, Real& B){
  detail::A_B_scalar_stable_ab(t,A,B);
}

template<class S>
PCC_ALWAYS_INLINE void inner_angles_from_orientation_generic(
  const Mat3<S>& R1, const Vec3<Real>& n_world, Real alpha,
  S &phi2_wrapped, S &theta2, Vec3<S> &z2_world, Mat3<S> &R2) noexcept
{
  Vec3<S> nprime{
    R1.a00*S(n_world.x) + R1.a10*S(n_world.y) + R1.a20*S(n_world.z),
    R1.a01*S(n_world.x) + R1.a11*S(n_world.y) + R1.a21*S(n_world.z),
    R1.a02*S(n_world.x) + R1.a12*S(n_world.y) + R1.a22*S(n_world.z)
  };
  S nrm = nprime.norm();
  if (valof(nrm) > 0) {
    S inv = S(1)/nrm; nprime = {nprime.x*inv, nprime.y*inv, nprime.z*inv};
  }
  S sa = S(std::sin(alpha)), ca = S(std::cos(alpha));
  S phi2 = (std::abs(std::sin(alpha)) < 1e-7) ? atan2(nprime.y, nprime.x) : atan2(nprime.y, nprime.x - sa);
  Mat3<S> Rz_minus = Mat3<S>::rotZ(-phi2);
  Vec3<S> n2 = Rz_minus.apply(nprime);
  Vec3<S> v  = Rz_minus.apply({ sa, S(0), ca });
  S vx=v.x, vz=v.z, nx=n2.x, nz=n2.z;
  S denom = vx*vx + vz*vz;
  S cos_th = (vx*nx + vz*nz) / denom;
  S sin_th = (vz*nx - vx*nz) / denom;
  S r = hypot(cos_th, sin_th);
  if (valof(r) > 0) { cos_th = cos_th / r; sin_th = sin_th / r; }
  theta2 = atan2(sin_th, cos_th);
  R2 = RzRyRz(phi2, theta2);
  z2_world = { R2.a02, R2.a12, R2.a22 };
  phi2_wrapped = wrap0_any(phi2);
}

template<class S>
struct FKGenOut{
  Vec3<S> end_no_d{}, end_with_d{}, b_world{};
  Mat3<S> R1{}, R2{}, Rtot{};
  S theta2{}, phi2{}, l2p{};
};

template<class S>
PCC_ALWAYS_INLINE FKGenOut<S> forward_exact_generic(
  S theta1, S phi1,
  const Vec3<Real>& P_star, const Vec3<Real>& n_star_in,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc)
{
  FKGenOut<S> o;
  Vec3<Real> n_star = Vec3<Real>{n_star_in.x,n_star_in.y,n_star_in.z}.normalized();
  S A1,B1; A_B_generic(theta1,A1,B1);
  auto sphi1 = sin(phi1), cphi1 = cos(phi1);
  Vec3<S> p_bend{ S(sc.s1)*A1*cphi1, S(sc.s1)*A1*sphi1, S(sc.s1)*B1 };
  Vec3<S> p1{S(0),S(0),S(sc.L1p)}; p1 = p1 + p_bend;
  o.R1 = RzRyRz(phi1, theta1);
  Real alpha = seg2.bevel_angle_deg * (std::numbers::pi_v<Real> / Real(180));
  Vec3<S> z2w; inner_angles_from_orientation_generic(o.R1, n_star, alpha, o.phi2, o.theta2, z2w, o.R2);
  S A2,B2; A_B_generic(o.theta2,A2,B2);
  Mat3<S> Rz2 = Mat3<S>::rotZ(o.phi2);
  Vec3<S> v2_local{ S(sc.s2_fixed)*A2, S(0), S(sc.s2_fixed)*B2 };
  Vec3<S> Rz2_v2 = Rz2.apply(v2_local);
  Vec3<S> R2_ez = o.R2.apply({S(0),S(0),S(1)});
  Vec3<S> dP{ S(P_star.x) - p1.x, S(P_star.y) - p1.y, S(P_star.z) - p1.z };
  Vec3<S> dP_local{
    o.R1.a00*dP.x + o.R1.a10*dP.y + o.R1.a20*dP.z,
    o.R1.a01*dP.x + o.R1.a11*dP.y + o.R1.a21*dP.z,
    o.R1.a02*dP.x + o.R1.a12*dP.y + o.R1.a22*dP.z
  };
  Real stub = std::max<Real>(Real(0), seg2.rigid_tip_length);
  S l2p_unc = dP_local.z - ( S(sc.s2_fixed)*B2 + S(stub)*R2_ez.z );
  Real lmin = seg2.passive_L_min, lmax = seg2.passive_L_max;
  if (lmax < lmin) { lmin = Real(0); lmax = Real(0); }
  o.l2p = clamp_any(l2p_unc, lmin, lmax);
  Vec3<S> pre{S(0),S(0),o.l2p};
  Vec3<S> tip_rigid = R2_ez * S(stub);
  Vec3<S> q_local = pre + Rz2_v2 + tip_rigid;
  Vec3<S> R1_q = o.R1.apply(q_local);
  Vec3<S> end_p_no_d{ p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z };
  o.end_no_d = end_p_no_d;
  S d_best = clamp_any(S(P_star.z) - end_p_no_d.z, sc.d_min, sc.d_max);
  o.end_with_d = { end_p_no_d.x, end_p_no_d.y, end_p_no_d.z + d_best };
  o.Rtot = o.R1.mul(o.R2);
  Vec3<S> b0{ S(std::sin(alpha)), S(0), S(std::cos(alpha)) };
  o.b_world = o.Rtot.apply(b0);
  return o;
}

template<class S>
PCC_ALWAYS_INLINE Vec3<S> residual_generic(S theta1, S phi1,
  const Vec3<Real>& P_star, const Vec3<Real>& n_star,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc)
{
  auto fk = forward_exact_generic<S>(theta1, phi1, P_star, n_star, seg1, seg2, sc);
  return { fk.end_with_d.x - S(P_star.x), fk.end_with_d.y - S(P_star.y), fk.end_with_d.z - S(P_star.z) };
}

static PCC_ALWAYS_INLINE Real theta1_seed_from_xy_radial(Real rho, Real s1, Real th_min, Real th_max){
  rho = std::max<Real>(0, rho);
  if (s1 <= Real(1e-9)) return std::clamp<Real>(0, th_min, th_max);
  const Real y = std::min<Real>(rho / s1, Real(0.99));
  Real t = std::clamp<Real>(Real(2)*y, std::max(th_min, Real(1e-4)), th_max);
  for (int i=0;i<6;++i){
    Real A,B,Ap,Bp; detail::A_B_scalar_stable_full(t,A,B,Ap,Bp);
    const Real f  = s1*A - rho;
    const Real fp = s1*Ap + Real(1e-8);
    t = std::clamp<Real>(t - f/fp, th_min, th_max);
  }
  return t;
}

struct FKOut{
  Vec3<Real> end_no_d{}, end_with_d{};
  Mat3<Real> R1{}, R2{}, Rtot{};
  Vec3<Real> b_world{};
  Real theta2{}, phi2{}, l2p{};
  bool feasible{true};
};

static PCC_ALWAYS_INLINE FKOut forward_exact(
  Real theta1, Real phi1,
  const Vec3<Real>& P_star, const Vec3<Real>& n_star_in,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc)
{
  set_thread_qos_interactive();
  auto fk = forward_exact_generic<Real>(theta1, phi1, P_star, n_star_in, seg1, seg2, sc);
  FKOut o;
  o.end_no_d=fk.end_no_d; o.end_with_d=fk.end_with_d;
  o.R1=fk.R1; o.R2=fk.R2; o.Rtot=fk.Rtot;
  o.b_world=fk.b_world;
  o.theta2=fk.theta2; o.phi2=fk.phi2; o.l2p=fk.l2p;
  return o;
}

struct Cand{
  Real pos_err{}, ang_err_deg{}, translation{}, abs_d{};
  Real theta1{}, phi1{}, theta2{}, phi2{}, l2p{};
  bool ok=false;
  Vec3<Real> end_p{}, b_world{};
};

static PCC_HOT PCC_FLATTEN Cand evaluate_once_core(
  Real theta1, Real phi1,
  const Vec3<Real>& P_star, const Vec3<Real>& n_star_in,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc)
{
  Cand out; out.ok=false;
  const Vec3<Real> n_star = n_star_in.normalized();
  const Real tol_th = Real(1e-4) * (std::numbers::pi_v<Real> / Real(180));
  if (!clamp_box(theta1, seg1.theta_min, seg1.theta_max, tol_th)) return out;
  phi1 = seg1.has_phi_bounds ? proj_ang(phi1, seg1.phi_min, seg1.phi_max) : wrap0(phi1);
  FKOut fk = forward_exact(theta1, phi1, P_star, n_star, seg1, seg2, sc);
  const Real cb = std::clamp(fk.b_world.dot(n_star), Real(-1), Real(1));
  const Real cos_bevel_tol = std::cos(sc.bevel_tol_deg * (std::numbers::pi_v<Real> / Real(180)));
  const Vec3<Real> diff{ fk.end_with_d.x - P_star.x, fk.end_with_d.y - P_star.y, fk.end_with_d.z - P_star.z };
  const Real pos_err2 = diff.dot(diff);
  const Real pos_tol2 = detail::sqr(sc.pos_tol) + Real(1e-12);
  if (pos_err2 > pos_tol2 || cb < cos_bevel_tol) return out;
  const Real ang_target = sc.angle_target_deg * (std::numbers::pi_v<Real> / Real(180));
  const Vec3<Real> ezw = fk.Rtot.apply({Real(0),Real(0),Real(1)});
  const Real cos_axis = std::clamp(ezw.dot(n_star), Real(-1), Real(1));
  const Real ang_err_deg = std::abs(std::acos(cos_axis) - ang_target) * Real(180)/std::numbers::pi_v<Real>;
  out.ok = true;
  out.pos_err = std::sqrt(pos_err2);
  out.ang_err_deg = ang_err_deg;
  out.translation = fk.end_with_d.z - fk.end_no_d.z;
  out.abs_d = std::abs(out.translation);
  out.theta1 = theta1; out.phi1 = wrap0(phi1);
  out.theta2 = fk.theta2; out.phi2 = wrap0(fk.phi2);
  out.l2p = fk.l2p;
  out.end_p = fk.end_with_d;
  out.b_world = fk.b_world;
  return out;
}

static void residual_and_jacobian(
  Real theta1, Real phi1,
  const Vec3<Real>& P_star, const Vec3<Real>& n_star,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc,
  Vec3<Real>& r, Real J[3][2])
{
  Dual2<Real> th(theta1,1,0), ph(phi1,0,1);
  auto rr = residual_generic<Dual2<Real>>(th, ph, P_star, n_star, seg1, seg2, sc);
  r = { rr.x.v, rr.y.v, rr.z.v };
  J[0][0]=rr.x.d0; J[0][1]=rr.x.d1;
  J[1][0]=rr.y.d0; J[1][1]=rr.y.d1;
  J[2][0]=rr.z.d0; J[2][1]=rr.z.d1;
}

static bool solve_outer_lm(
  const Vec3<Real>& P_star, const Vec3<Real>& n_star,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc,
  Real &theta1, Real &phi1)
{
  const Real rho = std::sqrt(P_star.x*P_star.x + P_star.y*P_star.y);
  phi1 = std::atan2(P_star.y, P_star.x);
  if (seg1.has_phi_bounds) phi1 = proj_ang(phi1, seg1.phi_min, seg1.phi_max);
  theta1 = theta1_seed_from_xy_radial(rho, sc.s1, seg1.theta_min, seg1.theta_max);
  {
    const Mat3<Real> R1s = RzRyRz(phi1, theta1);
    const Real alpha = seg2.bevel_angle_deg * (std::numbers::pi_v<Real> / Real(180));
    Real phi2, th2; Vec3<Real> z2w; Mat3<Real> R2;
    inner_angles_from_orientation_generic(R1s, n_star, alpha, phi2, th2, z2w, R2);
    Real A2,B2; detail::A_B_scalar_stable_ab(th2, A2, B2);
    const Mat3<Real> Rz2 = Mat3<Real>::rotZ(phi2);
    const Vec3<Real> v2_local{ sc.s2_fixed*A2, Real(0), sc.s2_fixed*B2 };
    const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
    const Vec3<Real> R2_ez = R2.apply({Real(0),Real(0),Real(1)});
    const Real stub = std::max<Real>(Real(0), seg2.rigid_tip_length);
    const Real est_xy = std::sqrt( detail::sqr(Rz2_v2.x + stub*R2_ez.x)
                                 + detail::sqr(Rz2_v2.y + stub*R2_ez.y) );
    const Real rho_eff = std::max<Real>(0, rho - est_xy);
    theta1 = theta1_seed_from_xy_radial(rho_eff, sc.s1, seg1.theta_min, seg1.theta_max);
  }
  const int maxit=20;
  Real lambda = 1e-3;
  for (int it=0; it<maxit; ++it){
    Real th_try=theta1, ph_try=phi1;
    Vec3<Real> r0; Real J[3][2];
    residual_and_jacobian(theta1, phi1, P_star, n_star, seg1, seg2, sc, r0, J);
    Real a00 = J[0][0]*J[0][0]+J[1][0]*J[1][0]+J[2][0]*J[2][0];
    Real a01 = J[0][0]*J[0][1]+J[1][0]*J[1][1]+J[2][0]*J[2][1];
    Real a11 = J[0][1]*J[0][1]+J[1][1]*J[1][1]+J[2][1]*J[2][1];
    Real g0  = J[0][0]*r0.x + J[1][0]*r0.y + J[2][0]*r0.z;
    Real g1  = J[0][1]*r0.x + J[1][1]*r0.y + J[2][1]*r0.z;
    Real r0n2 = r0.dot(r0);
    bool accepted=false;
    for (int retry=0; retry<2; ++retry){
      Real A00=a00+lambda, A01=a01, A11=a11+lambda;
      Real det = A00*A11 - A01*A01; if (std::abs(det)<1e-20) { lambda*=10; continue; }
      Real dth = (-A11*g0 + A01*g1)/det;
      Real dph = ( A01*g0 - A00*g1)/det;
      th_try = theta1 + dth; ph_try = phi1 + dph;
      const Real tol_th = Real(1e-4)*(std::numbers::pi_v<Real>/Real(180));
      if (!clamp_box(th_try, seg1.theta_min, seg1.theta_max, tol_th)) { lambda*=10; continue; }
      ph_try = seg1.has_phi_bounds ? proj_ang(ph_try, seg1.phi_min, seg1.phi_max) : wrap0(ph_try);
      auto fk = forward_exact(th_try, ph_try, P_star, n_star, seg1, seg2, sc);
      Vec3<Real> r1{ fk.end_with_d.x - P_star.x, fk.end_with_d.y - P_star.y, fk.end_with_d.z - P_star.z };
      Real r1n2 = r1.dot(r1);
      if (r1n2 < r0n2) { theta1=th_try; phi1=ph_try; lambda = std::max<Real>(1e-6, lambda*0.3); accepted=true; break; }
      lambda*=10;
    }
    if (!accepted) break;
    if (r0n2 < 1e-18) break;
  }
  return true;
}

static py::object solve_impl(
  py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
  py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc,
  int keep_top)
{
  auto P=P_star_np.unchecked<1>(); auto N=n_star_np.unchecked<1>();
  const Vec3<Real> P_star{P(0),P(1),P(2)}, n_star{N(0),N(1),N(2)};
  Real theta1=0, phi1=0;
  if (!solve_outer_lm(P_star, n_star, seg1, seg2, sc, theta1, phi1)) return py::none();
  Cand c = evaluate_once_core(theta1, phi1, P_star, n_star, seg1, seg2, sc);
  if (!c.ok) return py::none();
  py::list out; py::dict d;
  d["pos_err"]=c.pos_err; d["ang_err_deg"]=c.ang_err_deg;
  d["translation"]=c.translation; d["abs_d"]=c.abs_d;
  d["theta1"]=c.theta1; d["phi1"]=c.phi1;
  d["theta2"]=c.theta2; d["phi2"]=c.phi2; d["l2p"]=c.l2p;
  Real A1,B1; detail::A_B_scalar_stable_ab(c.theta1, A1,B1);
  Real sphi1,cphi1; detail::fast_sincos(c.phi1, sphi1,cphi1);
  const Vec3<Real> p_bend{ sc.s1*A1*cphi1, sc.s1*A1*sphi1, sc.s1*B1 };
  Vec3<Real> p1{0,0,sc.L1p}; p1 = p1 + p_bend;
  const Mat3<Real> R1 = RzRyRz(c.phi1, c.theta1);
  const Mat3<Real> R2 = RzRyRz(c.phi2, c.theta2);
  const Mat3<Real> Rtot = R1.mul(R2);
  Real A2,B2; detail::A_B_scalar_stable_ab(c.theta2, A2,B2);
  const Mat3<Real> Rz2 = Mat3<Real>::rotZ(c.phi2);
  const Vec3<Real> v2_local{ sc.s2_fixed*A2, Real(0), sc.s2_fixed*B2 };
  const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
  const Vec3<Real> R2_ez = R2.apply({Real(0),Real(0),Real(1)});
  const Real stub = std::max<Real>(Real(0), seg2.rigid_tip_length);
  const Vec3<Real> pre{0,0,c.l2p};
  const Vec3<Real> q_local{
    pre.x + Rz2_v2.x + stub*R2_ez.x,
    pre.y + Rz2_v2.y + stub*R2_ez.y,
    pre.z + Rz2_v2.z + stub*R2_ez.z
  };
  const Vec3<Real> R1_q = R1.apply(q_local);
  const Vec3<Real> end_p_no_d{ p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z };
  py::array_t<Real> T({4,4}); auto Tm=T.mutable_unchecked<2>();
  Tm(0,0)=Rtot.a00; Tm(0,1)=Rtot.a01; Tm(0,2)=Rtot.a02; Tm(0,3)=end_p_no_d.x;
  Tm(1,0)=Rtot.a10; Tm(1,1)=Rtot.a11; Tm(1,2)=Rtot.a12; Tm(1,3)=end_p_no_d.y;
  Tm(2,0)=Rtot.a20; Tm(2,1)=Rtot.a21; Tm(2,2)=Rtot.a22; Tm(2,3)=end_p_no_d.z;
  Tm(3,0)=0; Tm(3,1)=0; Tm(3,2)=0; Tm(3,3)=1;
  d["end_T"]=T;
  auto endw = py::array_t<Real>(3); auto ew=endw.mutable_unchecked<1>();
  ew(0)=c.end_p.x; ew(1)=c.end_p.y; ew(2)=c.end_p.z; d["end_p_world"]=endw;
  auto bw = py::array_t<Real>(3); auto b=bw.mutable_unchecked<1>();
  b(0)=c.b_world.x; b(1)=c.b_world.y; b(2)=c.b_world.z; d["bevel_world"]=bw;
  out.append(std::move(d)); return out;
}

static py::object solve_py(
  py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
  py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
  const SegmentBounds& seg1, const SegmentBounds& seg2, const SolverConst& sc,
  int keep_top = 1)
{
  return solve_impl(P_star_np, n_star_np, seg1, seg2, sc, keep_top);
}

PYBIND11_MODULE(_core, m) {
  py::class_<SegmentBounds>(m, "SegmentBounds")
    .def(py::init<>())
    .def_readwrite("theta_min", &SegmentBounds::theta_min)
    .def_readwrite("theta_max", &SegmentBounds::theta_max)
    .def_readwrite("has_phi_bounds", &SegmentBounds::has_phi_bounds)
    .def_readwrite("phi_min", &SegmentBounds::phi_min)
    .def_readwrite("phi_max", &SegmentBounds::phi_max)
    .def_readwrite("L_min", &SegmentBounds::L_min)
    .def_readwrite("L_max", &SegmentBounds::L_max)
    .def_readwrite("passive_L_min", &SegmentBounds::passive_L_min)
    .def_readwrite("passive_L_max", &SegmentBounds::passive_L_max)
    .def_readwrite("active_L_max", &SegmentBounds::active_L_max)
    .def_readwrite("bevel_angle_deg", &SegmentBounds::bevel_angle_deg)
    .def_readwrite("rigid_tip_length", &SegmentBounds::rigid_tip_length);

  py::class_<SolverConst>(m, "SolverConst")
    .def(py::init<>())
    .def_readwrite("s1", &SolverConst::s1)
    .def_readwrite("L1p", &SolverConst::L1p)
    .def_readwrite("s2_fixed", &SolverConst::s2_fixed)
    .def_readwrite("d_min", &SolverConst::d_min)
    .def_readwrite("d_max", &SolverConst::d_max)
    .def_readwrite("pos_tol", &SolverConst::pos_tol)
    .def_readwrite("bevel_tol_deg", &SolverConst::bevel_tol_deg)
    .def_readwrite("angle_target_deg", &SolverConst::angle_target_deg);

  m.def("solve", &solve_py,
        py::arg("P_star"), py::arg("n_star"),
        py::arg("seg1"), py::arg("seg2"), py::arg("sc"),
        py::arg("keep_top") = 1);
}
