"""2D tomograpgy example using entropy regularized optimal transport."""

import numpy as np
import odl
import time
import matplotlib.pyplot as plt
import sys

from utils import Logger, CallbackShowAndSave, CallbackPrintDiff
from transport_cost import EntropyRegularizedOptimalTransport, KMatrixFFT2


# Seed randomness for reproducability
np.random.seed(seed=1)


# =========================================================================== #
# Create a log that is written to disc
# =========================================================================== #
time_str = (str(time.localtime().tm_year) + str(time.localtime().tm_mon) +
            str(time.localtime().tm_mday) + '_' +
            str(time.localtime().tm_hour) + str(time.localtime().tm_min))
output_filename = 'Output_' + time_str + '.txt'

sys.stdout = Logger(output_filename)


# =========================================================================== #
# Set up the tomography problem and create phantom
# =========================================================================== #
# Select data type to use
dtype = 'float64'

# Create a discrete reconstruction space
n = 256

reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[n, n], dtype=dtype)

# Make a parallel beam geometry with flat detector with uniform angle and
# detector partition
angle_partition = odl.uniform_partition(np.pi/4, np.pi*3/4, 30)
detector_partition = odl.uniform_partition(-30, 30, 350)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True) + 1e-6

# =========================================================================== #
# Construct the prior
# =========================================================================== #
def rebin(a, shape):
    """Helper function for rebining.

    The following method for rebinning was found here:
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    sh = int(shape[0]), int(a.shape[0]//shape[0]), int(shape[1]), int(a.shape[1]//shape[1])
    return a.reshape(sh).mean(-1).mean(1)


# Define prior.
size_factor = 10
n_large = size_factor * n
high_dim_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                   shape=[n_large, n_large], dtype='float64')

tmp = (odl.phantom.shepp_logan(high_dim_space, modified=True) + 1e-6).asarray()

# Factors that determine the reshaping of the phantom
alpha0 = 1.05
shift0 = -0.03

alpha1 = 0.95
shift1 = 0.02

# Create distortion in the first dimensions
y0 = (np.round(n_large * shift0 + np.linspace(n_large*(1-1/alpha0)/2,
                                              n_large*(1+1/alpha0)/2,
                                              n_large))).astype('int64')
y0 = np.fmin(np.fmax(0, y0), n_large-1)

# Create distortion in the second dimensions
y1 = (np.round(n_large * shift1 + np.linspace(n_large*(1-1/alpha1)/2,
                                              n_large*(1+1/alpha1)/2,
                                              n_large))).astype('int64')
y1 = np.fmin(np.fmax(0, y1), n_large-1)

# Create a high-res. distortion and downsample it
tmp = tmp.take(y0, axis=0).take(y1, axis=1) / (alpha0 * alpha1)
tmp = rebin(tmp, (n_large/size_factor, n_large/size_factor))
prior = reco_space.element(tmp)

# Show the phantom and the prior
phantom.show(title='Phantom', saveto='Phantom')
prior.show(title='Prior', saveto='Prior')

no_title = '_no_title'
phantom.show(saveto='Phantom'+no_title)
prior.show(saveto='Prior'+no_title)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)


# =========================================================================== #
# Display to show how mass moves from the prior
# =========================================================================== #
ball_pos = [[1.5, 15.7], [16, -1], [2, -12.5]]
ball_rad = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]


def balls(x):
    """Helper function for drawing the circles."""
    ball = reco_space.zero()
    for i, r in zip(ball_pos, ball_rad):
        ball += reco_space.element(((x[0]-i[0])**2 +
                                    (x[1]-i[1])**2 <= r**2).astype(int))
    return ball

transport_mask = reco_space.element(balls)

fig_prior = prior.show()
ax_prior = fig_prior.gca()

for i, r in zip(ball_pos, ball_rad):
    circle = plt.Circle((i[0], i[1]), r, color='w', fill=False, linewidth=2.5)
    ax_prior.add_artist(circle)

fig_prior.savefig('masked_prior' + no_title)


# =========================================================================== #
# Parameters for the reconstructions
# =========================================================================== #
# Parameters for Douglas-Rachford solver
douglas_rachford_iter = 10000

scaling_tau = 0.05
scaling_sigma = 1.0 / scaling_tau
tau = 1.0 * scaling_tau  # tau same in all solvers. Some contains 3 sigmas.

scaling_tau_omt = 5.0
scaling_sigma_omt = 1.0 / scaling_tau_omt
tau_omt = 1.0 * scaling_tau_omt  # tau same in all solvers. Some contains 3 sigmas.

# Amount of added noise
noise_level = 0.05
data_eps = proj_data.norm() * noise_level * 1.2

# 1) TV reconstruction
reg_param_TV = 0.3

# 2) L2 + TV regularization with prior
reg_param_TV_l2_and_tv = 1.0

# 3) Optimal transport
sinkhorn_iter = 200
epsilon = 1.0
reg_param_op_TV = 1.0


# Add noise to data
noise = odl.phantom.white_noise(proj_data.space)
noise = noise / noise.norm() * proj_data.norm() * noise_level
noisy_data = proj_data + noise


# Constructing data-matching functional
data_func = odl.solvers.IndicatorLpUnitBall(proj_data.space, 2).translated(noisy_data / data_eps) / data_eps


# Components common for several methods
gradient = odl.Gradient(reco_space)
gradient_norm = odl.power_method_opnorm(gradient, maxiter=1000)

ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=1000)

show_func_L2_data = odl.solvers.L2Norm(proj_data.space).translated(noisy_data / data_eps) / data_eps * ray_trafo
show_func_TV = odl.solvers.GroupL1Norm(gradient.range) * gradient


# =========================================================================== #
# Print parameters used
# =========================================================================== #
print(reco_space)
print(geometry)

print('noise_level:', str(noise_level))
print('data_func =', str(data_func))

print('douglas_rachford_iter:' + str(douglas_rachford_iter))

print('scaling_tau: ', str(scaling_tau))
print('reg_param_TV_l2_and_tv: ', str(reg_param_TV_l2_and_tv))

print('scaling_tau_omt: ', str(scaling_tau_omt))
print('reg_param_op_TV: ', str(reg_param_op_TV))
print('sinkhorn_iter: ', str(sinkhorn_iter))
print('epsilon: ', str(epsilon))


# =========================================================================== #
# Settint up in order to print appropriate things in each iteration
# =========================================================================== #
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackPrintTiming() &
            CallbackShowAndSave(show_funcs=[show_func_L2_data, show_func_TV],
                                display_step=50) &
            CallbackPrintDiff(data_func=show_func_L2_data, display_step=2))


# =========================================================================== #
# Filtered Backprojection
# =========================================================================== #
# Create FBP reconstruction using a Hann filter
fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann',
                         frequency_scaling=0.7)

fbp_reconstruction = fbp_op(noisy_data)
fbp_reconstruction.show('Filtered backprojection',
                        saveto='Filtered Backprojection')
fbp_reconstruction.show(clim=[0, 1.0],
                        saveto='Filtered Backprojection'+no_title)


# =========================================================================== #
# TV
# =========================================================================== #
# Assemble TV functional
print('======================================================================')
print('TV')
print('======================================================================')
TV_func = reg_param_TV * odl.solvers.GroupL1Norm(gradient.range)
data_func_tv = data_func

f = odl.solvers.IndicatorBox(reco_space, lower=0)  # , upper=255)
g = [data_func_tv, TV_func]
L = [ray_trafo, gradient]
sigma_unscaled = [1 / ray_trafo_norm**2, 1 / gradient_norm**2]
sigma = [s * scaling_sigma for s in sigma_unscaled]

# Solve the problem
x_tv = reco_space.one()
callback.reset()


odl.solvers.douglas_rachford_pd(x=x_tv, f=f, g=g, L=L, tau=tau, sigma=sigma,
                                niter=douglas_rachford_iter, callback=callback)

x_tv.show('TV reconstruction', saveto='TV reconstruction')
x_tv.show(clim=[0, 1.0], saveto='TV reconstruction'+no_title)


# =========================================================================== #
# L2 + TV regularization with prior
# =========================================================================== #
# Assemble regularizing and data functional
print('======================================================================')
print('L2 + TV regularization with prior')
print('======================================================================')
for l2_reg_param_loop in [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]:
    print('=================================')
    print('l2_reg_param_loop: '+str(l2_reg_param_loop))
    print('=================================')

    data_func_l2_l2_and_tv = data_func
    l2_reg_func_l2_and_tv = l2_reg_param_loop * odl.solvers.L2NormSquared(
        reco_space).translated(prior)
    TV_func_l2_and_tv = reg_param_TV_l2_and_tv * odl.solvers.GroupL1Norm(gradient.range)

    f = odl.solvers.IndicatorBox(reco_space, lower=0)  # , upper=255)
    g = [data_func_l2_l2_and_tv, l2_reg_func_l2_and_tv, TV_func_l2_and_tv]
    L = [ray_trafo, odl.IdentityOperator(reco_space), gradient]
    sigma_unscaled = [1 / ray_trafo_norm**2, 1.0, 1 / gradient_norm**2]
    sigma = [s * scaling_sigma for s in sigma_unscaled]

    # Solve the problem
    x_l2 = reco_space.one()
    callback.reset()

    odl.solvers.douglas_rachford_pd(x=x_l2, f=f, g=g, L=L, tau=tau,
                                    sigma=sigma, niter=douglas_rachford_iter,
                                    callback=callback)

    x_l2.show(('L2 + TV regularization with prior reg param ' +
               str(l2_reg_param_loop)),
              saveto=('L2 plus TV regularization, reg param ' +
                      str(l2_reg_param_loop)).replace('.', '_'))
    x_l2.show(clim=[0, 1.0], saveto=('L2 plus TV regularization, reg param ' +
      str(l2_reg_param_loop).replace('.', '_') + no_title))


# =========================================================================== #
# Optimal transport
# =========================================================================== #
print('======================================================================')
print('Optimal transport:')
print('======================================================================')
# Define the transportation cost
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

for reg_para_loop in [4.0, 3.5]:
    print('=================================')
    print('reg_para_loop: ', str(reg_para_loop))
    print('=================================')
    try:
        # Creating the optimal transport functional and proximal
        opt_trans_func = EntropyRegularizedOptimalTransport(space=reco_space,
            matrix_param=matrix_param, K_class=KMatrixFFT2, epsilon=epsilon,
            mu0=prior, niter=sinkhorn_iter)

        callback_omt = (odl.solvers.CallbackPrintIteration() &
                        odl.solvers.CallbackPrintTiming() &
                        CallbackShowAndSave(file_prefix=('omt_dr_reg_' +
                                                         str(reg_para_loop) +
                                                         '_iter').replace('.',
                                                                          '_'),
                                            display_step=25,
                                            show_funcs=[show_func_L2_data, show_func_TV]) &
                        CallbackPrintDiff(data_func=show_func_L2_data,
                                          display_step=2))

        # Assemble data and TV functionals
        data_func_op = data_func
        TV_op_func = reg_param_op_TV * odl.solvers.GroupL1Norm(gradient.range)

        f = reg_para_loop * opt_trans_func
        g = [data_func_op, TV_op_func]
        L = [ray_trafo, gradient]
        sigma_unscaled_omt = [1/ray_trafo_norm**2, 1/gradient_norm**2]
        sigma_omt = [s * scaling_sigma_omt for s in sigma_unscaled_omt]

        # Solve the prolbem
        x_op = x_tv.copy() + 0.01  # Start from TV-reconstuction to save time
        callback.reset()

        t = time.time()  # Measure the time
        odl.solvers.douglas_rachford_pd(x=x_op, f=f, g=g, L=L, tau=tau_omt,
                                        lam=1.8, sigma=sigma_omt,
                                        niter=douglas_rachford_iter,
                                        callback=callback_omt)
        t = time.time() - t
        print('Time to solve the problem: {}'.format(int(t)))

        # Show and save reconstruction
        x_op.show(title=('Optimal transport + TV reconstruction, reg param ' +
                         str(reg_para_loop)),
                  saveto=('Optimal transport + TV reconstruction, reg param ' +
                          str(reg_para_loop)).replace('.', '_'))
        x_op.show(clim=[0, 1.0], saveto=('Optimal transport + TV reconstruction, reg param ' +
                                         str(reg_para_loop)).replace('.', '_') + no_title)

        # Show how mass moves from prior to reconstruction
        deformed_mask = opt_trans_func.deform_image(x_op, transport_mask)

        fig_defo_mask_text = deformed_mask.show('Mass movement from prior to reconstruction')
        fig_defo_mask = deformed_mask.show()

        ax_defo_mask_text = fig_defo_mask_text.gca()
        ax_defo_mask = fig_defo_mask.gca()

        for i, r in zip(ball_pos, ball_rad):
            circle_text = plt.Circle((i[0], i[1]), r, color='w', fill=False,
                                     linewidth=2.5)
            circle = plt.Circle((i[0], i[1]), r, color='w', fill=False,
                                linewidth=2.5)
            ax_defo_mask_text.add_artist(circle_text)
            ax_defo_mask.add_artist(circle)

        save_string = ('Optimal transport + TV reconstruction, ' +
                       'reg param ' + str(reg_para_loop).replace('.', '_') +
                       'mass movment')
        fig_defo_mask_text.savefig(save_string)
        fig_defo_mask.savefig(save_string + no_title)

    except:
        reco_space.one().show(saveto=('Crashed, reg param ' +
                                      str(reg_para_loop)).replace('.', '_') +
                              no_title)


# =========================================================================== #
# This dumps the omt reconstruction and other things to disc
# =========================================================================== #
import pickle
with open('omt_recon.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([prior, phantom, proj_data, noise, noisy_data, transport_mask,
                 x_op], f)

# Close the logger and only write in terminal again
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal