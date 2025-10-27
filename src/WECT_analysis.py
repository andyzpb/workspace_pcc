import time, math, random
import numpy as np
from pccik_native import _core as core

seg1 = core.SegmentBounds()
seg1.theta_min = math.radians(-135)
seg1.theta_max = math.radians( 135)
seg1.has_phi_bounds = False

seg2 = core.SegmentBounds()
seg2.theta_min = math.radians(-90)
seg2.theta_max = math.radians( 90)
seg2.has_phi_bounds = False
seg2.L_min = 0.0
seg2.L_max = 0.40
seg2.passive_L_min = 0.00
seg2.passive_L_max = 0.15
seg2.active_L_max  = 0.25
seg2.bevel_angle_deg = 45.0
seg2.rigid_tip_length = 0.02

sc = core.SolverConst()
sc.s1  = 0.08
sc.L1p = 0.02
sc.s2_fixed = 0.05
sc.d_min = -0.01
sc.d_max =  0.03
sc.pos_tol = 1e9
sc.bevel_tol_deg = 90.0
sc.angle_target_deg = 45.0

theta1 = math.radians(35.0)

P_star = np.array([0.03, -0.02, 0.12], dtype=np.float64)
n_star = np.array([0.6,  0.2,  0.77], dtype=np.float64)

def once(phi1):
    return core.evaluate_once(theta1, phi1, P_star, n_star, seg1, seg2, sc)

for _ in range(2000):
    once(random.uniform(0, 2*math.pi))

samples = []
N = 5000000
for _ in range(N):
    phi1 = random.uniform(0, 2*math.pi)
    t0 = time.perf_counter_ns()
    _ = once(phi1) 
    dt = time.perf_counter_ns() - t0
    samples.append(dt)

samples.sort()
def pct(p): return samples[int(min(len(samples)-1, p*len(samples)))]

print("count:", len(samples))
print("min/us  :", samples[0]/1e3)
print("p50/us  :", pct(0.50)/1e3)
print("p99.9/us:", pct(0.999)/1e3)
print("max/us  :", samples[-1]/1e3)
