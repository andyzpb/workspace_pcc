#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(__APPLE__)
#include <math.h> // for __sincos / __sincosf on macOS
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

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#ifndef PCC_REAL
#define PCC_REAL float // or float
#endif

using Real = PCC_REAL;

static inline void pcc_set_thread_qos_interactive() noexcept {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
#define PCC_ALWAYS_INLINE inline __attribute__((always_inline))
#define PCC_HOT __attribute__((hot))
#define PCC_COLD __attribute__((cold))
#define PCC_UNREACHABLE __builtin_unreachable()
#define PCC_ASSUME(cond)                                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      __builtin_unreachable();                                                 \
  } while (0)
#define PCC_FLATTEN __attribute__((flatten))
#else
#define PCC_ALWAYS_INLINE inline
#define PCC_HOT
#define PCC_COLD
#define PCC_UNREACHABLE __assume(0)
#define PCC_ASSUME(cond) __assume(cond)
#define PCC_FLATTEN
#endif

namespace detail {

template <class T> constexpr T sqr(T x) noexcept { return x * x; }

template <class T> inline constexpr T two_pi_v = T(2) * std::numbers::pi_v<T>;

template <class T> inline constexpr T one_over_two = T(0.5);

template <class T> inline constexpr T eps_v = std::numeric_limits<T>::epsilon();

template <class T>
PCC_ALWAYS_INLINE void fast_sincos(T a, T &s, T &c) noexcept {
#if (defined(__GNUC__) || defined(__clang__))
#if defined(__APPLE__)
  if constexpr (std::is_same_v<T, double>) {
    ::__sincos(a, &s, &c);
    return;
  }
  if constexpr (std::is_same_v<T, float>) {
    ::__sincosf(a, &s, &c);
    return;
  }
#else
  if constexpr (std::is_same_v<T, double>) {
    ::sincos(a, &s, &c);
    return;
  }
  if constexpr (std::is_same_v<T, float>) {
    ::sincosf(a, &s, &c);
    return;
  }
#endif
#endif
  s = std::sin(a);
  c = std::cos(a);
}

// Wrap angle into [0, 2π)
template <class T> PCC_ALWAYS_INLINE T wrap_0_2pi(T a) noexcept {
  const T twopi = two_pi_v<T>;
  const T inv_twopi = T(1) / twopi;
  T t = a - std::floor(a * inv_twopi) * twopi;
  if (t < T(0))
    t += twopi;
  if (t >= twopi)
    t -= twopi;
  return t;
}

// Project phi into interval [lo, hi] on the circle with minimal displacement
template <class T>
PCC_ALWAYS_INLINE T project_angle_to_interval(T phi, T lo, T hi) noexcept {
  lo = wrap_0_2pi(lo);
  hi = wrap_0_2pi(hi);
  const T twopi = two_pi_v<T>;
  const T span = std::fmod(hi - lo + twopi, twopi);
  if (span >= twopi - T(1e-12))
    return wrap_0_2pi(phi);

  T x = std::fmod(wrap_0_2pi(phi) - lo + twopi, twopi);
  if (x <= span)
    return lo + x;

  const T to_lo = std::fmod(twopi - x, twopi);
  const T to_hi = std::fmod(x - span, twopi);
  return (to_hi <= to_lo) ? hi : lo;
}

// Safe clamp to box with tolerance; returns false if "far out"
template <class T>
PCC_ALWAYS_INLINE bool snap_to_box(T &x, T lo, T hi, T tol) noexcept {
  if (x < lo - tol || x > hi + tol)
    return false;
  if (x < lo)
    x = lo;
  if (x > hi)
    x = hi;
  return true;
}

template <class T>
PCC_ALWAYS_INLINE void A_B_scalar_stable_ab(T t, T &A, T &B) noexcept {
  const T at = std::abs(t);
  // small threshold: 8 * cbrt(eps)
  const T small = T(8) * std::cbrt(eps_v<T>);

  if (at < small) [[likely]] {
    const T t2 = t * t;
    const T t3 = t2 * t;
    const T t4 = t2 * t2;
    const T t5 = t4 * t;
    const T t6 = t3 * t3;
    const T t7 = t6 * t;
    const T t8 = t4 * t4;

    // A ≈ t/2 - t^3/24 + t^5/720 - t^7/40320
    A = std::fma(
        T(-1.0 / 40320.0), t7,
        std::fma(T(1.0 / 720.0), t5, std::fma(T(-1.0 / 24.0), t3, T(0.5) * t)));

    // B ≈ 1 - t^2/6 + t^4/120 - t^6/5040 + t^8/362880
    B = std::fma(T(1.0 / 362880.0), t8,
                 std::fma(T(-1.0 / 5040.0), t6,
                          std::fma(T(1.0 / 120.0), t4,
                                   std::fma(T(-1.0 / 6.0), t2, T(1)))));
    return;
  }

  T s, c;
  fast_sincos(T(0.5) * t, s, c);
  const T s2 = s * s;
  // A = 2 s^2 / t
  A = (T(2) * s2) / t;
  // B = 2 s c / t
  B = (T(2) * s * c) / t;
}

template <class T>
PCC_ALWAYS_INLINE void A_B_scalar_stable(T t, T &A, T &B, T &Ap,
                                         T &Bp) noexcept {
  const T at = std::abs(t);
  const T small = T(8) * std::cbrt(eps_v<T>);

  if (at < small) [[likely]] {
    const T t2 = t * t;
    const T t3 = t2 * t;
    const T t4 = t2 * t2;
    const T t5 = t4 * t;
    const T t6 = t3 * t3;
    const T t7 = t6 * t;
    const T t8 = t4 * t4;

    A = std::fma(
        T(-1.0 / 40320.0), t7,
        std::fma(T(1.0 / 720.0), t5, std::fma(T(-1.0 / 24.0), t3, T(0.5) * t)));
    B = std::fma(T(1.0 / 362880.0), t8,
                 std::fma(T(-1.0 / 5040.0), t6,
                          std::fma(T(1.0 / 120.0), t4,
                                   std::fma(T(-1.0 / 6.0), t2, T(1)))));

    Ap = std::fma(
        T(-1.0 / 5760.0), t6,
        std::fma(T(1.0 / 144.0), t4, std::fma(T(-1.0 / 8.0), t2, T(0.5))));
    Bp = std::fma(T(1.0 / 45360.0), t7,
                  std::fma(T(-1.0 / 840.0), t5,
                           std::fma(T(1.0 / 30.0), t3, T(-1.0 / 3.0) * t)));
    return;
  }

  T s, c;
  fast_sincos(T(0.5) * t, s, c);

  const T s2 = s * s;
  A = (T(2) * s2) / t;
  B = (T(2) * s * c) / t;

  const T t2 = t * t;
  const T c2_minus_s2 = c * c - s2;
  Ap = (T(2) * s * c * t - T(2) * s2) / t2;
  Bp = (t * c2_minus_s2 - T(2) * s * c) / t2;
}

} // namespace detail

// ------------------------------ linear algebra
// -------------------------------------------
template <class T> struct [[nodiscard]] Vec3 {
  T x, y, z;

  PCC_ALWAYS_INLINE T dot(const Vec3 &o) const noexcept {
    return std::fma(x, o.x, std::fma(y, o.y, z * o.z));
  }
  PCC_ALWAYS_INLINE T norm2() const noexcept { return dot(*this); }
  PCC_ALWAYS_INLINE T norm() const noexcept { return std::sqrt(norm2()); }

  PCC_ALWAYS_INLINE Vec3 normalized(T eps = T(1e-12)) const noexcept {
    const T n = norm();
    if (n > eps) {
      const T inv = T(1) / n;
      return {x * inv, y * inv, z * inv};
    }
    return {T(0), T(0), T(0)};
  }

  PCC_ALWAYS_INLINE Vec3 operator+(const Vec3 &o) const noexcept {
    return {x + o.x, y + o.y, z + o.z};
  }
  PCC_ALWAYS_INLINE Vec3 operator-(const Vec3 &o) const noexcept {
    return {x - o.x, y - o.y, z - o.z};
  }
  PCC_ALWAYS_INLINE Vec3 operator*(T s) const noexcept {
    return {x * s, y * s, z * s};
  }
};

template <class T> struct [[nodiscard]] Mat3 {
  T a00, a01, a02, a10, a11, a12, a20, a21, a22;

  PCC_ALWAYS_INLINE Vec3<T> apply(const Vec3<T> &v) const noexcept {
    return {std::fma(a02, v.z, std::fma(a01, v.y, a00 * v.x)),
            std::fma(a12, v.z, std::fma(a11, v.y, a10 * v.x)),
            std::fma(a22, v.z, std::fma(a21, v.y, a20 * v.x))};
  }

  PCC_ALWAYS_INLINE Mat3 mul(const Mat3 &B) const noexcept {
    const Mat3 &A = *this;
    return {std::fma(A.a01, B.a10, std::fma(A.a02, B.a20, A.a00 * B.a00)),
            std::fma(A.a01, B.a11, std::fma(A.a02, B.a21, A.a00 * B.a01)),
            std::fma(A.a01, B.a12, std::fma(A.a02, B.a22, A.a00 * B.a02)),

            std::fma(A.a11, B.a10, std::fma(A.a12, B.a20, A.a10 * B.a00)),
            std::fma(A.a11, B.a11, std::fma(A.a12, B.a21, A.a10 * B.a01)),
            std::fma(A.a11, B.a12, std::fma(A.a12, B.a22, A.a10 * B.a02)),

            std::fma(A.a21, B.a10, std::fma(A.a22, B.a20, A.a20 * B.a00)),
            std::fma(A.a21, B.a11, std::fma(A.a22, B.a21, A.a20 * B.a01)),
            std::fma(A.a21, B.a12, std::fma(A.a22, B.a22, A.a20 * B.a02))};
  }

  static PCC_ALWAYS_INLINE Mat3 rotZ(T a) noexcept {
    T s, c;
    detail::fast_sincos(a, s, c);
    return {c, -s, 0, s, c, 0, 0, 0, 1};
  }
  static PCC_ALWAYS_INLINE Mat3 rotY(T a) noexcept {
    T s, c;
    detail::fast_sincos(a, s, c);
    return {c, 0, s, 0, 1, 0, -s, 0, c};
  }
};

// Closed form of Rz(phi) * Ry(theta) * Rz(-phi)
template <class T>
PCC_ALWAYS_INLINE Mat3<T> RzRyRz_compact(T phi, T theta) noexcept {
  T sphi, cphi;
  detail::fast_sincos(phi, sphi, cphi);
  T stheta, ctheta;
  detail::fast_sincos(theta, stheta, ctheta);

  const T cphi2 = cphi * cphi;
  const T sphi2 = sphi * sphi;
  const T cs = cphi * sphi;
  return {cphi2 * ctheta + sphi2, cs * (ctheta - T(1)),   cphi * stheta,
          cs * (ctheta - T(1)),   sphi2 * ctheta + cphi2, sphi * stheta,
          -cphi * stheta,         -sphi * stheta,         ctheta};
}

template <class T> PCC_ALWAYS_INLINE Mat3<T> RzRyRz(T phi, T theta) noexcept {
  return RzRyRz_compact<T>(phi, theta);
}

// ------------------------------ problem config
// -------------------------------------------
struct SegmentBounds {
  Real theta_min{}, theta_max{};
  bool has_phi_bounds{};
  Real phi_min{}, phi_max{};
  Real L_min{}, L_max{};
  Real passive_L_min{}, passive_L_max{};
  Real active_L_max{};
  Real bevel_angle_deg{Real(45)};
  Real rigid_tip_length{};
};

struct SolverConst {
  Real s1{}, L1p{};
  Real s2_fixed{};
  Real d_min{}, d_max{};
  Real pos_tol{Real(1e-6)};
  Real bevel_tol_deg{Real(1e-6)};
  Real angle_target_deg{Real(45)};
};

// ------------------------------ utilities (Real specialization)
// --------------------------
static inline Real wrap_0_2pi_R(Real a) { return detail::wrap_0_2pi(a); }
static inline Real project_angle_to_interval(Real phi, Real lo, Real hi) {
  return detail::project_angle_to_interval(phi, lo, hi);
}
static inline bool snap_to_box(Real &x, Real lo, Real hi, Real tol) {
  return detail::snap_to_box(x, lo, hi, tol);
}

// ------------------------------ geometry kernels
// -----------------------------------------
template <class T>
PCC_ALWAYS_INLINE void
inner_angles_from_orientation(const Mat3<T> &R1, const Vec3<T> &n_star, T alpha,
                              T &phi2_wrapped, T &theta2, Vec3<T> &z2_world,
                              Mat3<T> &R2) noexcept {
  // n' = R1^T * n*
  const Vec3<T> nprime{
      R1.a00 * n_star.x + R1.a10 * n_star.y + R1.a20 * n_star.z,
      R1.a01 * n_star.x + R1.a11 * n_star.y + R1.a21 * n_star.z,
      R1.a02 * n_star.x + R1.a12 * n_star.y + R1.a22 * n_star.z};

  T salpha, calpha;
  detail::fast_sincos(alpha, salpha, calpha);
  const T x = nprime.x - salpha, y = nprime.y;
  const T phi2 = std::atan2(y, x);

  T sph, cph;
  detail::fast_sincos(phi2, sph, cph);
  const T ux = salpha * cph, uz = calpha;
  const T wx = cph * nprime.x + sph * nprime.y;
  const T wz = nprime.z;

  const T den = std::max(ux * ux + uz * uz, std::numeric_limits<T>::epsilon());
  const T cos_th = (ux * wx + uz * wz) / den;
  const T sin_th = (uz * wx - ux * wz) / den;
  theta2 = std::atan2(sin_th, cos_th);

  constexpr T deg2rad = std::numbers::pi_v<T> / T(180);
  if (std::abs(theta2) < (T(0.5) * deg2rad)) [[likely]] {
    R2 = {T(1), 0, 0, 0, T(1), 0, 0, 0, T(1)};
    z2_world = {T(0), T(0), T(1)};
    phi2_wrapped = detail::wrap_0_2pi(phi2);
    theta2 = T(0);
    return;
  }

  R2 = RzRyRz(phi2, theta2);
  z2_world = {R2.a02, R2.a12, R2.a22};
  phi2_wrapped = detail::wrap_0_2pi(phi2);
}

struct Cand {
  Real pos_err{}, ang_err_deg{}, translation{}, abs_d{};
  Real theta1{}, phi1{}, theta2{}, phi2{}, L2p{};
  bool ok = false;
  Vec3<Real> end_p{}, b_world{};
};

static PCC_HOT PCC_FLATTEN Cand evaluate_once(Real theta1, Real phi1,
                                              const Vec3<Real> &P_star,
                                              const Vec3<Real> &n_star_in,
                                              const SegmentBounds &seg1,
                                              const SegmentBounds &seg2,
                                              const SolverConst &sc) noexcept {
  Cand out;
  out.ok = false;

  const Vec3<Real> n_star = n_star_in.normalized();

  const Real tol_th = Real(1e-4) * (std::numbers::pi_v<Real> / Real(180));
  const Real tol_th2 = Real(1e-4) * (std::numbers::pi_v<Real> / Real(180));

  if (!snap_to_box(theta1, seg1.theta_min, seg1.theta_max, tol_th))
    return out;

  if (seg1.has_phi_bounds)
    phi1 = project_angle_to_interval(phi1, seg1.phi_min, seg1.phi_max);
  else
    phi1 = wrap_0_2pi_R(phi1);

  const Mat3<Real> R1 = RzRyRz(phi1, theta1);

  Real A1, B1;
  detail::A_B_scalar_stable_ab(theta1, A1, B1);

  Real sphi1, cphi1;
  detail::fast_sincos(phi1, sphi1, cphi1);
  const Vec3<Real> p_bend{sc.s1 * A1 * cphi1, sc.s1 * A1 * sphi1, sc.s1 * B1};
  Vec3<Real> p1{Real(0), Real(0), sc.L1p};
  p1 = p1 + p_bend;

  const Real alpha =
      seg2.bevel_angle_deg * (std::numbers::pi_v<Real> / Real(180));
  const Real cos_bevel_tol =
      std::cos(sc.bevel_tol_deg * (std::numbers::pi_v<Real> / Real(180)));

  Mat3<Real> R2;
  Vec3<Real> z2w_unused;
  Real phi2, theta2;
  inner_angles_from_orientation(R1, n_star, alpha, phi2, theta2, z2w_unused,
                                R2);

  if (!snap_to_box(theta2, seg2.theta_min, seg2.theta_max, tol_th2))
    return out;

  if (seg2.has_phi_bounds) {
    if (!(seg2.phi_min - Real(1e-9) <= phi2 &&
          phi2 <= seg2.phi_max + Real(1e-9))) {
      const Real phi2_proj =
          project_angle_to_interval(phi2, seg2.phi_min, seg2.phi_max);
      if (phi2_proj != phi2) {
        phi2 = phi2_proj;
        R2 = RzRyRz(phi2, theta2);
      }
    }
  }

  Real A2, B2;
  detail::A_B_scalar_stable_ab(theta2, A2, B2);
  const Vec3<Real> v2_local{sc.s2_fixed * A2, Real(0), sc.s2_fixed * B2};

  const Mat3<Real> Rz2 = Mat3<Real>::rotZ(phi2);
  const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
  const Vec3<Real> R2_ez = R2.apply({Real(0), Real(0), Real(1)});

  const Vec3<Real> dP{P_star.x - p1.x, P_star.y - p1.y, P_star.z - p1.z};
  const Vec3<Real> dP_local{R1.a00 * dP.x + R1.a10 * dP.y + R1.a20 * dP.z,
                            R1.a01 * dP.x + R1.a11 * dP.y + R1.a21 * dP.z,
                            R1.a02 * dP.x + R1.a12 * dP.y + R1.a22 * dP.z};

  Real L2p = dP_local.z - seg2.rigid_tip_length * R2_ez.z - sc.s2_fixed * B2;

  const Real tol_L = Real(5e-4);
  const Real L2p_lo = seg2.passive_L_min;
  const Real L2p_hi = seg2.passive_L_max;
  if (L2p < L2p_lo - tol_L || L2p > L2p_hi + tol_L) [[unlikely]]
    return out;
  if (L2p < L2p_lo)
    L2p = L2p_lo;
  if (L2p > L2p_hi)
    L2p = L2p_hi;

  const Real L2_total_max = seg2.L_max;
  if (sc.s2_fixed + L2p > L2_total_max + tol_L) {
    if (L2_total_max < sc.s2_fixed - tol_L) [[unlikely]]
      return out;
    L2p = std::max(L2p_lo, std::min(L2p_hi, L2_total_max - sc.s2_fixed));
  }

  const Vec3<Real> q_no_L2p{Rz2_v2.x + seg2.rigid_tip_length * R2_ez.x,
                            Rz2_v2.y + seg2.rigid_tip_length * R2_ez.y,
                            Rz2_v2.z + seg2.rigid_tip_length * R2_ez.z};
  const Vec3<Real> q_local{q_no_L2p.x, q_no_L2p.y, q_no_L2p.z + L2p};
  const Vec3<Real> R1_q = R1.apply(q_local);
  const Vec3<Real> end_p_no_d{p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z};

  const Real d_best = std::clamp(P_star.z - end_p_no_d.z, sc.d_min, sc.d_max);
  const Vec3<Real> end_p{end_p_no_d.x, end_p_no_d.y, end_p_no_d.z + d_best};

  const Mat3<Real> R1R2 = R1.mul(R2);
  const Vec3<Real> b0{std::sin(alpha), Real(0), std::cos(alpha)};
  const Vec3<Real> b_world = R1R2.apply(b0);

  const Real cb = std::clamp(b_world.dot(n_star), Real(-1), Real(1));

  const Vec3<Real> diff{end_p.x - P_star.x, end_p.y - P_star.y,
                        end_p.z - P_star.z};
  const Real pos_err2 = diff.dot(diff);
  const Real pos_tol2 = detail::sqr(sc.pos_tol) + Real(1e-12);
  if (pos_err2 > pos_tol2 || cb < cos_bevel_tol) [[unlikely]]
    return out;
  const Real pos_err = std::sqrt(pos_err2);

  const Vec3<Real> ezw = R1R2.apply({Real(0), Real(0), Real(1)});
  const Real cos_axis = std::clamp(ezw.dot(n_star), Real(-1), Real(1));
  const Real ang_err_deg =
      std::abs(std::acos(cos_axis) * Real(180) / std::numbers::pi_v<Real> -
               sc.angle_target_deg);

  out.ok = true;
  out.pos_err = pos_err;
  out.ang_err_deg = ang_err_deg;
  out.translation = d_best;
  out.abs_d = std::abs(d_best);
  out.theta1 = theta1;
  out.phi1 = wrap_0_2pi_R(phi1);
  out.theta2 = theta2;
  out.phi2 = wrap_0_2pi_R(phi2);
  out.L2p = L2p;
  out.end_p = end_p;
  out.b_world = b_world;
  return out;
}

static std::pair<Real, Cand>
brent_refine_phi_native(Real theta1, Real phi_seed, Real halfspan_deg,
                        int maxiter, const Vec3<Real> &P_star,
                        const Vec3<Real> &n_star, const SegmentBounds &seg1,
                        const SegmentBounds &seg2, const SolverConst &sc) {
  pcc_set_thread_qos_interactive();
  const Real span = halfspan_deg * (std::numbers::pi_v<Real> / Real(180));
  Real a = phi_seed - span, b = phi_seed + span;

  struct ScoreFn {
    Real theta1;
    const Vec3<Real> &P_star;
    const Vec3<Real> &n_star;
    const SegmentBounds &seg1;
    const SegmentBounds &seg2;
    const SolverConst &sc;

    PCC_ALWAYS_INLINE Real operator()(Real ph) const noexcept {
      auto c = evaluate_once(theta1, ph, P_star, n_star, seg1, seg2, sc);
      if (!c.ok)
        return Real(1e9);
      return c.pos_err + Real(1e-6) * c.ang_err_deg;
    }
  } score{theta1, P_star, n_star, seg1, seg2, sc};

  const Real invphi = (std::sqrt(Real(5)) - Real(1)) / Real(2);
  Real x = a + invphi * (b - a), w = x, v = x;
  Real fx = score(x), fw = fx, fv = fx;
  Real d = 0, e = 0;
  const Real tol = Real(2e-6);
  constexpr Real eps = std::numeric_limits<Real>::epsilon();

  for (int it = 0; it < maxiter; ++it) {
    const Real m = (a + b) * Real(0.5);
    const Real tol1 = std::sqrt(eps) * std::abs(x) + tol / Real(3);
    const Real tol2 = tol1 * Real(2);
    if (std::abs(x - m) <= tol2 - Real(0.5) * (b - a))
      break;

    Real p = 0, q = 0, r = 0;
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = Real(2) * (q - r);
      if (q > 0)
        p = -p;
      q = std::abs(q);

      if (std::abs(p) < std::abs(Real(0.5) * q * e) && p > q * (a - x) &&
          p < q * (b - x)) {
        d = p / q;
        const Real u = x + d;
        if ((u - a) < tol2 || (b - u) < tol2)
          d = (x < m) ? tol1 : -tol1;
      } else {
        e = (x >= m) ? (a - x) : (b - x);
        d = invphi * e;
      }
    } else {
      e = (x >= m) ? (a - x) : (b - x);
      d = invphi * e;
    }
    const Real u = x + (std::abs(d) >= tol1 ? d : (d > 0 ? tol1 : -tol1));
    const Real fu = score(u);

    if (fu <= fx) {
      if (u < x)
        b = x;
      else
        a = x;
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x)
        a = u;
      else
        b = u;
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  auto best = evaluate_once(theta1, x, P_star, n_star, seg1, seg2, sc);
  return {x, best};
}

static std::pair<Vec3<Real>, std::pair<Real, Real>>
lm_polish_native(Real theta1, Real phi1, int iters, const Vec3<Real> &P_star,
                 const Vec3<Real> &n_star, const SegmentBounds &seg1,
                 const SegmentBounds &seg2, const SolverConst &sc,
                 Real &out_pos_err, Real &out_ang_err, Real &out_abs_d,
                 Real &best_th, Real &best_ph) {
  pcc_set_thread_qos_interactive();
  const Real tol_th = Real(1e-4) * (std::numbers::pi_v<Real> / Real(180));
  const Real wb = Real(1e-6);
  Real lam = Real(1e-2);
  const Real sqrt_eps = std::sqrt(std::numeric_limits<Real>::epsilon());

  auto residual = [&](Real th, Real ph, Vec3<Real> &endp,
                      Vec3<Real> &bev) -> std::array<Real, 6> {
    auto c = evaluate_once(th, ph, P_star, n_star, seg1, seg2, sc);
    if (!c.ok) {
      endp = {0, 0, 0};
      bev = {0, 0, 0};
      return {Real(1e3), Real(1e3), Real(1e3), Real(1e2), Real(1e2), Real(1e2)};
    }
    endp = c.end_p;
    bev = c.b_world;
    const Vec3<Real> rp{P_star.x - endp.x, P_star.y - endp.y,
                        P_star.z - endp.z};
    const Vec3<Real> bn = bev.normalized();
    const Vec3<Real> rb{n_star.x - bn.x, n_star.y - bn.y, n_star.z - bn.z};
    const Real sw = std::sqrt(wb);
    return {rp.x, rp.y, rp.z, sw * rb.x, sw * rb.y, sw * rb.z};
  };

  auto objective = [&](Real th, Real ph) -> Real {
    auto c = evaluate_once(th, ph, P_star, n_star, seg1, seg2, sc);
    if (!c.ok)
      return Real(1e9);
    return c.pos_err + Real(1e-9) * c.ang_err_deg + Real(1e-7) * c.abs_d;
  };

  Real th = theta1, ph = phi1;
  Real f_best = objective(th, ph);
  best_th = th;
  best_ph = ph;

  for (int it = 0; it < iters; ++it) {
    Vec3<Real> endp{}, bev{};
    const auto r = residual(th, ph, endp, bev);

    // Scale-aware finite differences
    const Real hth = sqrt_eps * std::max<Real>(1, std::abs(th));
    const Real hph = sqrt_eps * std::max<Real>(1, std::abs(ph));

    std::array<Real, 6> rp, rm;
    Real Jth[6]{}, Jph[6]{};

    {
      Vec3<Real> e1{}, b1{}, e2{}, b2{};
      rp = residual(th + hth, ph, e1, b1);
      rm = residual(th - hth, ph, e2, b2);
    }
    for (int i = 0; i < 6; ++i)
      Jth[i] = (rp[i] - rm[i]) / (Real(2) * hth);

    {
      Vec3<Real> e1{}, b1{}, e2{}, b2{};
      rp = residual(th, ph + hph, e1, b1);
      rm = residual(th, ph - hph, e2, b2);
    }
    for (int i = 0; i < 6; ++i)
      Jph[i] = (rp[i] - rm[i]) / (Real(2) * hph);

    Real a00 = 0, a01 = 0, a11 = 0, g0 = 0, g1 = 0;
    for (int i = 0; i < 6; ++i) {
      a00 += Jth[i] * Jth[i];
      a01 += Jth[i] * Jph[i];
      a11 += Jph[i] * Jph[i];
      g0 += -Jth[i] * r[i];
      g1 += -Jph[i] * r[i];
    }
    a00 += lam;
    a11 += lam;

    const Real det = a00 * a11 - a01 * a01;
    Real s0 = 0, s1 = 0;
    if (std::abs(det) > Real(1e-20)) {
      s0 = (a11 * g0 - a01 * g1) / det;
      s1 = (-a01 * g0 + a00 * g1) / det;
    }

    Real th_try = th + s0;
    if (!snap_to_box(th_try, seg1.theta_min, seg1.theta_max, tol_th)) {
      lam *= Real(2);
      continue;
    }
    Real ph_try = ph + s1;
    ph_try = seg1.has_phi_bounds
                 ? project_angle_to_interval(ph_try, seg1.phi_min, seg1.phi_max)
                 : wrap_0_2pi_R(ph_try);

    const Real f_try = objective(th_try, ph_try);
    if (f_try < f_best) {
      th = th_try;
      ph = ph_try;
      f_best = f_try;
      best_th = th;
      best_ph = ph;
      lam *= Real(0.5);
    } else {
      lam *= Real(2);
    }
  }

  const auto cfin =
      evaluate_once(best_th, best_ph, P_star, n_star, seg1, seg2, sc);
  out_pos_err = cfin.ok ? cfin.pos_err : Real(1e9);
  out_ang_err = cfin.ok ? cfin.ang_err_deg : Real(1e9);
  out_abs_d = cfin.ok ? cfin.abs_d : Real(1e9);
  return {cfin.end_p, {best_th, best_ph}};
}

// ------------------------------ Python wrappers
// ------------------------------------------
static py::object evaluate_once_py(
    Real theta1, Real phi1,
    py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
    py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
    const SegmentBounds &seg1, const SegmentBounds &seg2,
    const SolverConst &sc) {
  auto P = P_star_np.unchecked<1>();
  auto N = n_star_np.unchecked<1>();
  const Vec3<Real> P_star{P(0), P(1), P(2)}, n_star{N(0), N(1), N(2)};

  auto c = evaluate_once(theta1, phi1, P_star, n_star, seg1, seg2, sc);
  if (!c.ok)
    return py::none();

  const Mat3<Real> R1 = RzRyRz(c.phi1, c.theta1);

  Real A1, B1;
  detail::A_B_scalar_stable_ab(c.theta1, A1, B1);

  Real sphi1, cphi1;
  detail::fast_sincos(c.phi1, sphi1, cphi1);
  const Vec3<Real> p_bend{sc.s1 * A1 * cphi1, sc.s1 * A1 * sphi1, sc.s1 * B1};
  Vec3<Real> p1{Real(0), Real(0), sc.L1p};
  p1 = p1 + p_bend;

  const Mat3<Real> R2 = RzRyRz(c.phi2, c.theta2);

  Real A2, B2;
  detail::A_B_scalar_stable_ab(c.theta2, A2, B2);
  const Vec3<Real> v2_local{sc.s2_fixed * A2, Real(0), sc.s2_fixed * B2};
  const Mat3<Real> Rz2 = Mat3<Real>::rotZ(c.phi2);
  const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
  const Vec3<Real> R2_ez = R2.apply({Real(0), Real(0), Real(1)});

  const Vec3<Real> q_local{Rz2_v2.x + seg2.rigid_tip_length * R2_ez.x,
                           Rz2_v2.y + seg2.rigid_tip_length * R2_ez.y,
                           Rz2_v2.z + seg2.rigid_tip_length * R2_ez.z + c.L2p};
  const Vec3<Real> R1_q = R1.apply(q_local);
  const Vec3<Real> end_p_no_d{p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z};

  const Mat3<Real> Rtot = R1.mul(R2);
  py::array_t<Real> T({4, 4});
  auto Tm = T.mutable_unchecked<2>();
  Tm(0, 0) = Rtot.a00;
  Tm(0, 1) = Rtot.a01;
  Tm(0, 2) = Rtot.a02;
  Tm(0, 3) = end_p_no_d.x;
  Tm(1, 0) = Rtot.a10;
  Tm(1, 1) = Rtot.a11;
  Tm(1, 2) = Rtot.a12;
  Tm(1, 3) = end_p_no_d.y;
  Tm(2, 0) = Rtot.a20;
  Tm(2, 1) = Rtot.a21;
  Tm(2, 2) = Rtot.a22;
  Tm(2, 3) = end_p_no_d.z;
  Tm(3, 0) = 0;
  Tm(3, 1) = 0;
  Tm(3, 2) = 0;
  Tm(3, 3) = 1;

  py::dict d;
  d["pos_err"] = c.pos_err;
  d["ang_err_deg"] = c.ang_err_deg;
  d["translation"] = c.translation;
  d["abs_d"] = c.abs_d;
  d["theta1"] = c.theta1;
  d["phi1"] = c.phi1;
  d["theta2"] = c.theta2;
  d["phi2"] = c.phi2;
  d["L2p"] = c.L2p;

  auto endp0 = py::array_t<Real>(3);
  auto e0 = endp0.mutable_unchecked<1>();
  e0(0) = end_p_no_d.x;
  e0(1) = end_p_no_d.y;
  e0(2) = end_p_no_d.z;
  d["end_p"] = endp0;

  auto endp = py::array_t<Real>(3);
  auto e = endp.mutable_unchecked<1>();
  e(0) = c.end_p.x;
  e(1) = c.end_p.y;
  e(2) = c.end_p.z;
  d["end_p_world"] = endp;

  auto bw = py::array_t<Real>(3);
  auto b = bw.mutable_unchecked<1>();
  b(0) = c.b_world.x;
  b(1) = c.b_world.y;
  b(2) = c.b_world.z;
  d["bevel_world"] = bw;

  d["end_T"] = T;
  return d;
}

template <class T> struct TopK {
  int k;
  std::vector<T> v;
  explicit TopK(int kk) : k(kk) { v.reserve(std::max(kk * 3, 16)); }
  PCC_ALWAYS_INLINE void push(T &&t) { v.emplace_back(std::move(t)); }
  void shrink() {
    if ((int)v.size() <= k)
      return;
    std::nth_element(v.begin(), v.begin() + k, v.end(),
                     [](const T &a, const T &b) { return a.J < b.J; });
    v.resize(k);
  }
  void finalize() {
    shrink();
    std::sort(v.begin(), v.end(),
              [](const T &a, const T &b) { return a.J < b.J; });
  }
};

static py::list
phi_scan(Real theta1,
         py::array_t<Real, py::array::c_style | py::array::forcecast> phi_list,
         py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
         py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
         const SegmentBounds &seg1, const SegmentBounds &seg2,
         const SolverConst &sc, int k_keep = 6) {
  auto P = P_star_np.unchecked<1>();
  auto N = n_star_np.unchecked<1>();
  const Vec3<Real> P_star{P(0), P(1), P(2)}, n_star{N(0), N(1), N(2)};
  auto ph = phi_list.unchecked<1>();

  struct Score {
    Real J;
    Cand c;
  };
  std::vector<Score> merged;
  merged.reserve(std::max(k_keep * 4, 64));

  {
    py::gil_scoped_release nogil;

    auto worker = [&](int begin, int end) {
      pcc_set_thread_qos_interactive();
      TopK<Score> tk(k_keep);
      for (int i = begin; i < end; ++i) {
        auto cand =
            evaluate_once(theta1, ph(i), P_star, n_star, seg1, seg2, sc);
        if (!cand.ok)
          continue;
        const Real cb = std::clamp(cand.b_world.dot(n_star), Real(-1), Real(1));
        const Real J = cand.pos_err + Real(1e-3) * (Real(1) - cb) +
                       Real(1e-7) * cand.abs_d;

        tk.push(Score{J, std::move(cand)});
      }
      tk.finalize();
      return tk.v;
    };

#if defined(USE_OPENMP)
    const int n = (int)ph.shape(0);
    const int T = std::max(1, omp_get_max_threads());
    const int chunk = (n + T - 1) / T;
    std::vector<std::vector<Score>> buckets(T);
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int b = tid * chunk, e = std::min(n, b + chunk);
      buckets[tid] = worker(b, e);
    }
    for (auto &vv : buckets) {
      merged.insert(merged.end(), std::make_move_iterator(vv.begin()),
                    std::make_move_iterator(vv.end()));
    }
#else
    auto vv = worker(0, (int)ph.shape(0));
    merged.insert(merged.end(), std::make_move_iterator(vv.begin()),
                  std::make_move_iterator(vv.end()));
#endif
  }

  if ((int)merged.size() > k_keep) {
    std::nth_element(merged.begin(), merged.begin() + k_keep, merged.end(),
                     [](const Score &a, const Score &b) { return a.J < b.J; });
    merged.resize(k_keep);
  }
  std::sort(merged.begin(), merged.end(),
            [](const Score &a, const Score &b) { return a.J < b.J; });

  py::list out;
  for (auto &s : merged) {
    const auto &c = s.c;
    py::dict d;
    d["pos_err"] = c.pos_err;
    d["ang_err_deg"] = c.ang_err_deg;
    d["translation"] = c.translation;
    d["abs_d"] = c.abs_d;
    d["theta1"] = c.theta1;
    d["phi1"] = c.phi1;
    d["theta2"] = c.theta2;
    d["phi2"] = c.phi2;
    d["L2p"] = c.L2p;
    out.append(std::move(d));
  }
  return out;
}

static py::object brent_refine_phi_py(
    Real theta1, Real phi_seed, Real halfspan_deg, int maxiter,
    py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
    py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
    const SegmentBounds &seg1, const SegmentBounds &seg2,
    const SolverConst &sc) {
  auto P = P_star_np.unchecked<1>();
  auto N = n_star_np.unchecked<1>();
  const Vec3<Real> P_star{P(0), P(1), P(2)}, n_star{N(0), N(1), N(2)};

  Real phi_best;
  Cand cand;
  {
    py::gil_scoped_release nogil;
    std::tie(phi_best, cand) =
        brent_refine_phi_native(theta1, phi_seed, halfspan_deg, maxiter, P_star,
                                n_star, seg1, seg2, sc);
  }

  if (!cand.ok)
    return py::none();

  const Mat3<Real> R1 = RzRyRz(cand.phi1, cand.theta1);

  Real A1, B1;
  detail::A_B_scalar_stable_ab(cand.theta1, A1, B1);
  Real sphi1, cphi1;
  detail::fast_sincos(cand.phi1, sphi1, cphi1);
  const Vec3<Real> p_bend{sc.s1 * A1 * cphi1, sc.s1 * A1 * sphi1, sc.s1 * B1};
  Vec3<Real> p1{Real(0), Real(0), sc.L1p};
  p1 = p1 + p_bend;

  const Mat3<Real> R2 = RzRyRz(cand.phi2, cand.theta2);
  Real A2, B2;
  detail::A_B_scalar_stable_ab(cand.theta2, A2, B2);
  const Vec3<Real> v2_local{sc.s2_fixed * A2, Real(0), sc.s2_fixed * B2};
  const Mat3<Real> Rz2 = Mat3<Real>::rotZ(cand.phi2);
  const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
  const Vec3<Real> R2_ez = R2.apply({Real(0), Real(0), Real(1)});

  const Vec3<Real> q_local{Rz2_v2.x + seg2.rigid_tip_length * R2_ez.x,
                           Rz2_v2.y + seg2.rigid_tip_length * R2_ez.y,
                           Rz2_v2.z + seg2.rigid_tip_length * R2_ez.z +
                               cand.L2p};
  const Vec3<Real> R1_q = R1.apply(q_local);
  const Vec3<Real> end_p_no_d{p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z};

  const Mat3<Real> Rtot = R1.mul(R2);
  py::array_t<Real> T({4, 4});
  auto Tm = T.mutable_unchecked<2>();
  Tm(0, 0) = Rtot.a00;
  Tm(0, 1) = Rtot.a01;
  Tm(0, 2) = Rtot.a02;
  Tm(0, 3) = end_p_no_d.x;
  Tm(1, 0) = Rtot.a10;
  Tm(1, 1) = Rtot.a11;
  Tm(1, 2) = Rtot.a12;
  Tm(1, 3) = end_p_no_d.y;
  Tm(2, 0) = Rtot.a20;
  Tm(2, 1) = Rtot.a21;
  Tm(2, 2) = Rtot.a22;
  Tm(2, 3) = end_p_no_d.z;
  Tm(3, 0) = 0;
  Tm(3, 1) = 0;
  Tm(3, 2) = 0;
  Tm(3, 3) = 1;

  py::dict d;
  d["phi1"] = phi_best;
  d["pos_err"] = cand.pos_err;
  d["ang_err_deg"] = cand.ang_err_deg;
  d["translation"] = cand.translation;
  d["abs_d"] = cand.abs_d;
  d["theta1"] = cand.theta1;
  d["phi1_eval"] = cand.phi1;

  auto endp0 = py::array_t<Real>(3);
  auto e0 = endp0.mutable_unchecked<1>();
  e0(0) = end_p_no_d.x;
  e0(1) = end_p_no_d.y;
  e0(2) = end_p_no_d.z;
  d["end_p"] = endp0;
  d["end_T"] = T;

  auto endw = py::array_t<Real>(3);
  auto ew = endw.mutable_unchecked<1>();
  ew(0) = cand.end_p.x;
  ew(1) = cand.end_p.y;
  ew(2) = cand.end_p.z;
  d["end_p_world"] = endw;

  auto bw = py::array_t<Real>(3);
  auto b = bw.mutable_unchecked<1>();
  b(0) = cand.b_world.x;
  b(1) = cand.b_world.y;
  b(2) = cand.b_world.z;
  d["bevel_world"] = bw;

  return d;
}

static py::object lm_polish_py(
    Real theta1, Real phi1, int iters,
    py::array_t<Real, py::array::c_style | py::array::forcecast> P_star_np,
    py::array_t<Real, py::array::c_style | py::array::forcecast> n_star_np,
    const SegmentBounds &seg1, const SegmentBounds &seg2,
    const SolverConst &sc) {
  auto P = P_star_np.unchecked<1>();
  auto N = n_star_np.unchecked<1>();
  const Vec3<Real> P_star{P(0), P(1), P(2)}, n_star{N(0), N(1), N(2)};

  Real pos_err, ang_err, abs_d, th_fin, ph_fin;
  {
    py::gil_scoped_release nogil;
    (void)lm_polish_native(theta1, phi1, iters, P_star, n_star, seg1, seg2, sc,
                           pos_err, ang_err, abs_d, th_fin, ph_fin);
  }

  const auto cfin =
      evaluate_once(th_fin, ph_fin, P_star, n_star, seg1, seg2, sc);
  if (!cfin.ok) {
    py::dict d;
    d["theta1"] = th_fin;
    d["phi1"] = ph_fin;
    d["pos_err"] = pos_err;
    d["ang_err_deg"] = ang_err;
    d["abs_d"] = abs_d;
    return d;
  }

  const Mat3<Real> R1 = RzRyRz(cfin.phi1, cfin.theta1);

  Real A1, B1;
  detail::A_B_scalar_stable_ab(cfin.theta1, A1, B1);
  Real sphi1, cphi1;
  detail::fast_sincos(cfin.phi1, sphi1, cphi1);
  const Vec3<Real> p_bend{sc.s1 * A1 * cphi1, sc.s1 * A1 * sphi1, sc.s1 * B1};
  Vec3<Real> p1{Real(0), Real(0), sc.L1p};
  p1 = p1 + p_bend;

  const Mat3<Real> R2 = RzRyRz(cfin.phi2, cfin.theta2);
  Real A2, B2;
  detail::A_B_scalar_stable_ab(cfin.theta2, A2, B2);
  const Vec3<Real> v2_local{sc.s2_fixed * A2, Real(0), sc.s2_fixed * B2};
  const Mat3<Real> Rz2 = Mat3<Real>::rotZ(cfin.phi2);
  const Vec3<Real> Rz2_v2 = Rz2.apply(v2_local);
  const Vec3<Real> R2_ez = R2.apply({Real(0), Real(0), Real(1)});

  const Vec3<Real> q_local{Rz2_v2.x + seg2.rigid_tip_length * R2_ez.x,
                           Rz2_v2.y + seg2.rigid_tip_length * R2_ez.y,
                           Rz2_v2.z + seg2.rigid_tip_length * R2_ez.z +
                               cfin.L2p};
  const Vec3<Real> R1_q = R1.apply(q_local);
  const Vec3<Real> end_p_no_d{p1.x + R1_q.x, p1.y + R1_q.y, p1.z + R1_q.z};

  const Mat3<Real> Rtot = R1.mul(R2);
  py::array_t<Real> T({4, 4});
  auto Tm = T.mutable_unchecked<2>();
  Tm(0, 0) = Rtot.a00;
  Tm(0, 1) = Rtot.a01;
  Tm(0, 2) = Rtot.a02;
  Tm(0, 3) = end_p_no_d.x;
  Tm(1, 0) = Rtot.a10;
  Tm(1, 1) = Rtot.a11;
  Tm(1, 2) = Rtot.a12;
  Tm(1, 3) = end_p_no_d.y;
  Tm(2, 0) = Rtot.a20;
  Tm(2, 1) = Rtot.a21;
  Tm(2, 2) = Rtot.a22;
  Tm(2, 3) = end_p_no_d.z;
  Tm(3, 0) = 0;
  Tm(3, 1) = 0;
  Tm(3, 2) = 0;
  Tm(3, 3) = 1;

  py::dict d;
  d["theta1"] = th_fin;
  d["phi1"] = ph_fin;
  d["pos_err"] = pos_err;
  d["ang_err_deg"] = ang_err;
  d["abs_d"] = abs_d;

  auto endp0 = py::array_t<Real>(3);
  auto e0 = endp0.mutable_unchecked<1>();
  e0(0) = end_p_no_d.x;
  e0(1) = end_p_no_d.y;
  e0(2) = end_p_no_d.z;
  d["end_p"] = endp0;
  d["end_T"] = T;

  auto endw = py::array_t<Real>(3);
  auto ew = endw.mutable_unchecked<1>();
  ew(0) = cfin.end_p.x;
  ew(1) = cfin.end_p.y;
  ew(2) = cfin.end_p.z;
  d["end_p_world"] = endw;

  auto bw = py::array_t<Real>(3);
  auto b = bw.mutable_unchecked<1>();
  b(0) = cfin.b_world.x;
  b(1) = cfin.b_world.y;
  b(2) = cfin.b_world.z;
  d["bevel_world"] = bw;

  return d;
}

// ------------------------------ pybind11 module
// ------------------------------------------
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

  m.def("phi_scan", &phi_scan, py::arg("theta1"), py::arg("phi_list"),
        py::arg("P_star"), py::arg("n_star"), py::arg("seg1"), py::arg("seg2"),
        py::arg("sc"), py::arg("k_keep") = 6);

  m.def("evaluate_once", &evaluate_once_py, py::arg("theta1"), py::arg("phi1"),
        py::arg("P_star"), py::arg("n_star"), py::arg("seg1"), py::arg("seg2"),
        py::arg("sc"));

  m.def("brent_refine_phi", &brent_refine_phi_py, py::arg("theta1"),
        py::arg("phi_seed"), py::arg("halfspan_deg"), py::arg("maxiter"),
        py::arg("P_star"), py::arg("n_star"), py::arg("seg1"), py::arg("seg2"),
        py::arg("sc"));

  m.def("lm_polish", &lm_polish_py, py::arg("theta1"), py::arg("phi1"),
        py::arg("iters"), py::arg("P_star"), py::arg("n_star"), py::arg("seg1"),
        py::arg("seg2"), py::arg("sc"));
}
