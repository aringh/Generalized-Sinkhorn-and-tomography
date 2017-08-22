"""The entropy-regularized optimal transport functional."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
from numpy.fft import (fft2, ifft2)

import scipy

from odl.solvers.functional.functional import Functional
from odl.operator import Operator
from odl.space.base_ntuples import FnBase


# =========================================================================== #
# Classes and functions related to entropy-regularized optimal transport
# =========================================================================== #

def lambertw_fulfix(x):
    """Helper class for stabilizing computations of the Lambert W function."""
    space = x.space
    tmp = scipy.special.lambertw(x).asarray().astype('float').flatten()

    # For all indices that returned NaN, make an approximation.
    ind1 = np.where(np.isnan(tmp) == True)
    tmp[ind1] = np.log(x[ind1] - np.log(x[ind1]))

    return space.element(np.reshape(tmp, space.shape))


# TODO: current implementation only works when the two marginals belong to the
# same space.
class EntropyRegularizedOptimalTransport(Functional):

    """The entropy regularized optimal transport functional.

    The functional is given by::

        T_eps(x) = min_M trace(C^T * M) + sum_{i,j} (m_{i,j} * log(m_{i,j}) -
              m_{i,j} + 1)
        subject to mu0 = M * 1
                   x = M^T * 1

    where ``C`` is the cost for transportation, ``M`` is the transportation
    plan, ``mu0`` is a given marginal, and ``x`` is the second marginal which
    is considered a free variable in this setting.
    """

    def __init__(self, space, matrix_param, epsilon, mu0, K_class=None,
                 niter=100):
        """Initialize a new instance.

        Parameters
        ----------
        space : ``DiscreteLp`` or ``FnBase``
            Domain of the functional.
        matrix_param : array
            Matrix parametrization of the transportation cost ``C`` compatible
            with the ``K_class``.
        epsilon : positive float
            Regularization parameter in the transportation cost.
        mu0 : ``space`` ``element-like``
            NOTE THAT WE WANT TRANSPORT BETWEEN DIFFERENT SPACES!
            HOW TO DO THIS?
        K_class : ``Operator``
            Operator whos action represents the multiplication with the matrix
            ``K``.
            Default: ``KFullMatrix``.
        niter : positive integer
            Number of iterations in Sinkhorn iterations in order to evaluate
            the functional and the proximal.
            Default: 100.
        """
        super().__init__(space=space, linear=False)

        self.__mu0 = mu0
        self.__niter = niter
        self.__epsilon = epsilon
        self.__matrix_param = matrix_param

        if K_class is None:
            self.__K_class = KFullMatrix
        else:
            self.__K_class = K_class

        self.__K_op = self.__K_class(np.exp(-self.matrix_param/self.epsilon),
                                     domain=self.domain, range=self.domain)

        self.__K_op_adjoint = self.K_op.adjoint
        self.__CK_op = self.__K_class(
            self.matrix_param * np.exp(-self.matrix_param/self.epsilon),
            domain=self.domain, range=self.domain)

        self.__tmp_u = self.domain.element()
        self.__tmp_v = self.domain.element()
        self.__tmp_x = self.domain.element()

        self.__tmp_u_prox = self.domain.one()
        self.__tmp_v_prox = self.domain.one()

    @property
    def matrix_param(self):
        """The parameterization of the matrix."""
        return self.__matrix_param

    @property
    def mu0(self):
        """The given margin to match in the transportation."""
        return self.__mu0

    @property
    def epsilon(self):
        """The regularization parameter in regularized optimal transport."""
        return self.__epsilon

    @property
    def niter(self):
        """Number of iterations in Sinkhorn to evaluate functional."""
        return self.__niter

    @property
    def K_op(self):
        """The K-operator (matrix)."""
        return self.__K_op

    @property
    def K_op_adjoint(self):
        """The adjoint of the K-operator (matrix)."""
        return self.__K_op_adjoint

    @property
    def CK_op(self):
        """The CK-operator (matrix)."""
        return self.__CK_op

    # Getters and setters for som temporary internal variables coming from the
    # prox-computations
    @property
    def tmp_u_prox(self):
        return self.__tmp_u_prox

    @tmp_u_prox.setter
    def tmp_u_prox(self, value):
        self.__tmp_u_prox = value

    @property
    def tmp_v_prox(self):
        return self.__tmp_v_prox

    @tmp_v_prox.setter
    def tmp_v_prox(self, value):
        self.__tmp_v_prox = value

    def _call(self, x):
        """Return the value of the functional."""
        # Running the Sinkhorn iterations
        u, v = self.return_diagonal_scalings(x)

        return (u.inner(self.CK_op(v)) +
                self.epsilon * (u * np.log(u)).inner(self.K_op(v)) +
                self.epsilon * u.inner(self.K_op(v * np.log(v))) -
                u.inner(self.CK_op(v)) +
                self.epsilon * self.domain.one().norm()**2 -
                self.epsilon * self.domain.one().inner(self.mu0))

    def return_diagonal_scalings(self, x):
        """Performs the Sinkhorn iterations and returns the two vecotrs used
        for the diagonal scaling."""
        u = self.domain.element()
        v = self.domain.one()

        # Running the Sinkhorn iterations
        for j in range(self.niter):
            tmp = np.fmax(self.K_op(v), 1e-30)
            u = self.mu0 / tmp
            tmp = np.fmax(self.K_op_adjoint(u), 1e-30)
            v = x / tmp

        self.__tmp_u = u
        self.__tmp_v = v
        self.__tmp_x = x
        return u, v

    def deform_image(self, x, mask):
        """Return..."""
        if x == self.__tmp_x:
            u = self.__tmp_u
            v = self.__tmp_v
        else:
            u, v = self.return_diagonal_scalings(x)

        tmp1 = self.K_op_adjoint(u*mask)
        tmp2 = np.fmax(tmp1, 1e-30)
        res = v * tmp2

        return res

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return NotImplemented

    @property
    def proximal(self):
        """Return the proximal factory of the functional."""
        functional = self

        class EntRegOptTransProximal(Operator):

            """Proximal operator of entropy regularized optimal transport.

            The prox is given by::

                prox_[gamma*T_eps](mu1) = arg min_x (T_epsilon(mu0, x) +
                                              1/(2*gamma) ||x - mu1||^2_2)
            """

            def __init__(self, sigma):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                """
                self.sigma = float(sigma)
                super().__init__(domain=functional.domain,
                                 range=functional.domain, linear=False)

                # Setting up parameters
                self.const = 1 / (functional.epsilon * sigma)

            def _call(self, x):
                """Apply the operator to ``x``."""
                u = functional.tmp_u_prox
                v = functional.tmp_v_prox

                # Running generalized Sinkhorn iterations
                for j in range(functional.niter):
                    # Safe-guarded u-update, to avoid divide-by-zero error.
                    u_old = u.copy()
                    tmp1 = functional.K_op(v)
                    if np.min(tmp1) < 1e-30 or np.max(tmp1) > 1e+50:
                        print('Numerical instability, truncation in Transport prox (Kv)',
                              str(np.min(tmp1)), str(np.max(tmp1)))

                    tmp = np.fmax(tmp1, 1e-30)


                    u = functional.mu0 / tmp
                    if np.min(u) < 1e-30 or np.max(u) > 1e+50:
                        print('u (min/max)', str(np.min(u)), str(np.max(u)))

                    # Safe-guarded v-update, to avoid divide-by-zero error.
                    v_old = v.copy()

                    tmp3 = functional.K_op_adjoint(u)
                    if np.min(tmp3) < 1e-30 or np.max(tmp3) > 1e+50:
                        print('Truncation in Transport prox (KTu)',
                              str(np.min(tmp3)), str(np.max(tmp3)))
                        print('u (min/max)', str(np.min(u)), str(np.max(u)))

                    tmp4 = (self.const * tmp3 * np.exp(self.const * x))

                    if np.min(tmp4) < 1e-30 or np.max(tmp4) > 1e+200:
                        print('Argument in lambdert omega (min/max)',
                              str(np.min(tmp4)), str(np.max(tmp4)))

                    v = np.exp(self.const * x - lambertw_fulfix(tmp4))

                    v1 = np.exp(self.const * x - scipy.special.lambertw(
                        tmp4))
                    if (v-v1).norm() > 1e-10:
                        print('diff pga ny lambderw omega funciton',
                              str((v-v1).norm()))
                        print('v (min/max)', str(np.min(v)), str(np.max(v)))
                        print('Argument in lambdert omega (min/max)',
                              str(np.min(tmp4)), str(np.max(tmp4)))

                    # If the updates in both u and v are small, break the loop
                    if ((np.log(v)-np.log(v_old)).norm() < 1e-8 and
                            (np.log(u)-np.log(u_old)).norm() < 1e-8):
                        break

                # Store the u and v in the internal temporary variables of the
                # functional
                functional.tmp_u_prox = u
                functional.tmp_v_prox = v

                return x - self.sigma * functional.epsilon * np.log(v)

        return EntRegOptTransProximal


# TODO: Matrix argument is named different things in the two different classes
class KFullMatrix(Operator):

    """The K-operator to use in Sinkhorn iterations, defined by a matrix.

    This is a linear operator corrsponding to the K-matrix/operator in the
    Sinkhorn iterations. This operator is created by giving the operator the
    full transportation cost matrix.
    """

    def __init__(self, cost_matrix, domain, range):
        """Initialize a new instance.

        Parameters
        ----------
        cost_matrix : `array-like` or  `scipy.sparse.spmatrix`
            Matrix representing the linear operator. Its shape must be
            ``(m, n)``, where ``n`` is the size of ``domain`` and ``m`` the
            size of ``range``. Its dtype must be castable to the range
            ``dtype``.
        domain : `DiscreteLp` or `FnBase`
            Space on whose elements the matrix acts.
        range : `DiscreteLp` or `FnBase`
            Space to which the matrix maps.
        """
        self.__cost_matrix = np.asarray(cost_matrix)

        if self.cost_matrix.ndim != 2:
            raise ValueError('matrix {} has {} axes instead of 2'
                             ''.format(cost_matrix, self.cost_matrix.ndim))

        if not isinstance(domain, FnBase):
            raise TypeError('`domain` {!r} is not an `FnBase` instance'
                            ''.format(domain))

        if not isinstance(range, FnBase):
            raise TypeError('`range` {!r} is not an `FnBase` instance'
                            ''.format(range))

        # Check compatibility of matrix with domain and range
        if not np.can_cast(domain.dtype, range.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}'
                            ''.format(domain.dtype, range.dtype))

        if self.cost_matrix.shape != (range.size, domain.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}'
                             ''.format(self.cost_matrix.shape,
                                       (range.size, domain.size),
                                       domain, range))

        if not np.can_cast(self.cost_matrix.dtype, range.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(cost_matrix.dtype, range.dtype))

        super().__init__(domain, range, linear=True)

    @property
    def cost_matrix(self):
        """The matrix defining the cost for the optimal transport."""
        return self.__cost_matrix

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix."""
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {})'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return KFullMatrix(self.cost_matrix.conj().T,
                           domain=self.range, range=self.domain)

    def _call(self, x):
        """Apply the operator to a point ``x``."""
        tmp = x.asarray().flatten()
        return self.range.element(self.cost_matrix.dot(tmp))


class KMatrixFFT2(Operator):

    """The K-operator to use in Sinkhorn iterations, defined by a matrix.

    This is a linear operator corrsponding to the K-matrix/operator in the
    Sinkhorn iterations. It use FFT to compute the matrix vector product, and
    does not store the entire cost matrix explicitly.
    """

    def __init__(self, dist_matrix, domain, range):
        """Initialize a new instance.

        Parameters
        ----------
        dist_matrix : `array-like` or  `scipy.sparse.spmatrix`
            Matrix representing the distance to the other pixels from the to
            left corner. Its shape must be as ...?.
            Its dtype must be castable to the range ``dtype``.
        domain : `DiscreteLp` or `FnBase`
            Space on whose elements the matrix acts.
        range : `DiscreteLp` or `FnBase`
            Space to which the matrix maps.
        """
        self.__dist_matrix = np.asarray(dist_matrix)

        if self.dist_matrix.ndim != 2:
            raise ValueError('matrix {} has {} axes instead of 2'
                             ''.format(dist_matrix, self.dist_matrix.ndim))

        if not isinstance(domain, FnBase):
            raise TypeError('`domain` {!r} is not an `FnBase` instance'
                            ''.format(domain))

        if not isinstance(range, FnBase):
            raise TypeError('`range` {!r} is not an `FnBase` instance'
                            ''.format(range))

        # Check compatibility of matrix with domain and range
        if not np.can_cast(domain.dtype, range.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}'
                            ''.format(domain.dtype, range.dtype))

        if not np.can_cast(self.dist_matrix.dtype, range.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(dist_matrix.dtype, range.dtype))

        super().__init__(domain, range, linear=True)

        self.__n1, self.__n2 = self.dist_matrix.shape
        self.__dist_matrix_fft = fft2(np.pad(self.dist_matrix,
                                             ((0, self.__n1-1),
                                              (0, self.__n2-1)),
                                             'symmetric'))

    @property
    def dist_matrix(self):
        """The distance matrix, which defines the cost."""
        return self.__dist_matrix

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix."""
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {})'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return KMatrixFFT2(self.dist_matrix.conj().T,
                           domain=self.range, range=self.domain)

    def _call(self, x):
        """Apply the operator to a point ``x``."""
        x_ext_fft = fft2(np.pad(x.asarray(),
                                ((0, self.__n1-1), (0, self.__n2-1)),
                                'constant'))

        return ifft2(
         self.__dist_matrix_fft * x_ext_fft)[:self.__n1, :self.__n2]
