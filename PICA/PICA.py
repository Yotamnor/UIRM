import numpy as np

samples_num = 1000
final_dim = 1

Z_I_1 = np.random.randn(1, samples_num)
Z_I_2 = np.random.randn(1, samples_num)

Z_1 = np.sqrt(10)*np.random.randn(1, samples_num)
Z_2 = np.sqrt(2)*np.random.randn(1, samples_num)

eps_1 = 0.05*np.random.randn(3, samples_num)
eps_2 = 0.05*np.random.randn(3, samples_num)

A_I = np.array([[1], [1], [1]])
A_e = np.array([[1], [1], [-1]])

X_1 = A_I@Z_I_1 + A_e@Z_1 + eps_1
X_2 = A_I@Z_I_2 + A_e@Z_2 + eps_2

# X_1 = X_1.T
# X_2 = X_2.T

mu_1 = np.mean(X_1, axis=1)
mu_2 = np.mean(X_2, axis=1)

Sigma_1 = np.cov(X_1)
Sigma_2 = np.cov(X_2)

Sigma_s = Sigma_1-Sigma_2
Sigma_a = (Sigma_1+Sigma_2)/2
#
# lam1, U1 = np.linalg.eig(Sigma_1)
# lam2, U2 = np.linalg.eig(Sigma_2)

lam_s, U_s = np.linalg.eig(Sigma_s)
# lam_a, U_a = np.linalg.eig(Sigma_a)

b_lamd = np.max(lam_s)
U = np.zeros(Sigma_s.shape)
for i in range(lam_s.shape[0]):
    if lam_s[i] > (b_lamd/100):
        continue
    U[:,i] = U_s[:,i]

non_zero_columns = ~np.all(U == 0, axis=0)
U_sq = U[:, non_zero_columns]

Sigma_null = (U_sq.T)@(Sigma_a@U_sq)
lam_null, V = np.linalg.eig(Sigma_null)

idx = np.argsort(lam_null)[::-1]

lam_null = lam_null[idx]
V = V[:, idx]

U_r = np.zeros((lam_s.shape[0], final_dim))
for j in range(final_dim):
    U_r[:,j] = U_sq@(V[:,j])

import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
fig, axes = plt.subplots(1, 3, figsize=(25, 7), subplot_kw={'projection': '3d'})
ax0 = fig.add_subplot(131, projection='3d')
# Extract the individual components from X_1 and X_2
X_1_x, X_1_y, X_1_z = X_1
X_2_x, X_2_y, X_2_z = X_2

# Plot X_1 in blue
ax0.scatter(X_1_x, X_1_y, X_1_z+10, c='blue', label='X_1 Samples', alpha=0.3)
# Plot X_2 in red
ax0.scatter(X_2_x, X_2_y, X_2_z, c='red', label='X_2 Samples', alpha=0.3)

# Labels and legend
ax0.set_xlabel("X-axis")
ax0.set_ylabel("Y-axis")
ax0.set_zlabel("Z-axis")
ax0.set_title("Original Data")
ax0.legend()

# plt.show()

# Reduction time!

ax1 = fig.add_subplot(132)

X_1_d1 = (U_r.T)@X_1
X_2_d1 = (U_r.T)@X_2

std_x1 = np.std(X_1_d1)
std_x2 = np.std(X_2_d1)

# Create an array for positioning (since scatter needs x & y values)
indices_x1 = np.ones_like(X_1_d1)  # Align x1 at y=0
indices_x2 = np.zeros_like(X_2_d1)    # Align x2 at y=1
# Plot the scatter points
ax1.scatter(X_1_d1, indices_x1, c='blue', label=f'x1 (Std: {std_x1:.2f})', alpha=0.6)
ax1.scatter(X_2_d1, indices_x2, c='red', label=f'x2 (Std: {std_x2:.2f})', alpha=0.6)

# Formatting
ax1.set_xlabel("Values")
ax1.set_ylabel("Environments")
ax1.set_title("Projection Over the Principal Invariant Direction")
ax1.set_yticks([0, 1], ["x1", "x2"])  # Label y-axis
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
X_1_d3 = (U_r)@X_1_d1
X_2_d3 = (U_r)@X_2_d1

ax2 = fig.add_subplot(133, projection='3d')

# Extract the individual components from X_1 and X_2
X_1_x, X_1_y, X_1_z = X_1_d3
X_2_x, X_2_y, X_2_z = X_2_d3

# Plot X_1 in blue
ax2.scatter(X_1_x, X_1_y, X_1_z+10, c='blue', label='X_1 Samples', alpha=0.3)
# Plot X_2 in red
ax2.scatter(X_2_x, X_2_y, X_2_z, c='red', label='X_2 Samples', alpha=0.3)

# Labels and legend
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")
ax2.set_zlabel("Z-axis")
ax2.set_title("Back to the Original Space")
ax2.legend()

fig
plt.show()
