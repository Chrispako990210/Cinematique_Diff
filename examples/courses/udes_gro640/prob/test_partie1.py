import numpy as np
import pacc2101 as pacc

# Test 1

# Create main
q = np.zeros(5)
q[0] = np.pi/2
q[1] = 0
q[2] = 0
q[3] = 0
q[4] = 0

r = pacc.f(q)
print("q = q0, q1, q2, q3, q4 : ", q)
print("r = x, y, z : ", r)
