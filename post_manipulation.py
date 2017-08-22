"""Short script for post-manipulation of the images for the hand-example. In
order to show how the mass is moved from prior to final reconstruction.
"""

import numpy as np
import odl
import matplotlib.pyplot as plt
import pickle

from transport_cost import EntropyRegularizedOptimalTransport, KMatrixFFT2

with open('omt_recon.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    prior, phantom, proj_data, noise, noisy_data, transport_mask, x_op = pickle.load(f)


# =========================================================================== #
# Create the same space as in the hand-example (could maybe pickle this...)
# =========================================================================== #
# Data type to use
dtype = 'float64'

# Discrete reconstruction space
n = 256
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[n, n], dtype=dtype)


# String to save images with no title
no_title = '_no_title'


# =========================================================================== #
# Display to show how mass moves
# =========================================================================== #
ball_pos = [[10, 2], [-5, 5], [2, -7]]
ball_rad = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]


def balls(x):
    ball = reco_space.zero()
    for i, r in zip(ball_pos, ball_rad):
        ball += reco_space.element(((x[0]-i[0])**2 +
                                    (x[1]-i[1])**2 <= r**2).astype(int))
    return ball


# =========================================================================== #
# Same parameters as when reconstructing
# =========================================================================== #
sinkhorn_iter = 200
epsilon = 1.5


# =========================================================================== #
# Create the same transport cost and compute the movement
# =========================================================================== #
tmp = np.arange(0, n, 1, dtype=dtype) * (1 / n) * 40.0  # Normalize cost to n indep.

tmp = tmp[:, np.newaxis]
v_ones = np.ones(n, dtype=dtype)
v_ones = v_ones[np.newaxis, :]
x = np.dot(tmp, v_ones)

tmp = np.transpose(tmp)
v_ones = np.transpose(v_ones)
y = np.dot(v_ones, tmp)

tmp_mat = (x + 1j*y).flatten()
tmp_mat = tmp_mat[:, np.newaxis]
long_v_ones = np.transpose(np.ones(tmp_mat.shape, dtype=dtype))

# This is the matrix defining the distance
matrix_param = np.minimum(20.0**2, np.abs(x + 1j*y)**2)

# The reg-parameter used. Just to save with correct name
reg_para_loop = 4.0

# Creating the optimal transport functional
opt_trans_func = EntropyRegularizedOptimalTransport(space=reco_space,
                                                    matrix_param=matrix_param,
                                                    K_class=KMatrixFFT2,
                                                    epsilon=epsilon,
                                                    mu0=prior-1e-4+1e-3,  # Slightly bigger lift of prior
                                                    niter=sinkhorn_iter)


# =========================================================================== #
# Post manipulation of the transport
# =========================================================================== #
# Make the recinstruction slightly more well-conditioned
tmp = np.min(x_op)
x_op_ture = x_op.copy()
if tmp < 0:
    x_op = x_op + (1e-3 - tmp)
elif tmp < 1e-3:
    x_op = x_op + 1e-3

# Show deformation and save the images
deformed_mask = opt_trans_func.deform_image(x_op, transport_mask)

fig_defo_mask_text = deformed_mask.show('Mass movement from prior to reconstruction')
fig_defo_mask = deformed_mask.show()

ax_defo_mask_text = fig_defo_mask_text.gca()
ax_defo_mask = fig_defo_mask.gca()

for i, r in zip(ball_pos, ball_rad):
    circle_text = plt.Circle((i[0], i[1]), r, color='w', fill=False, linewidth=2.5)
    circle = plt.Circle((i[0], i[1]), r, color='w', fill=False, linewidth=2.5)
    ax_defo_mask_text.add_artist(circle_text)
    ax_defo_mask.add_artist(circle)

save_string = ('Optimal transport + TV reconstruction, ' +
               'reg param ' + str(reg_para_loop).replace('.', '_') +
               'mass movment_postManipulation')
fig_defo_mask_text.savefig(save_string)
fig_defo_mask.savefig(save_string + no_title)
