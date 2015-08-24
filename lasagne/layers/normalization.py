# -*- coding: utf-8 -*-

"""
The :class:`LocalResponseNormalization2DLayer
<lasagne.layers.LocalResponseNormalization2DLayer>` implementation contains
code from `pylearn2 <http://github.com/lisa-lab/pylearn2>`_, which is covered
by the following license:


Copyright (c) 2011--2014, Université de Montréal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import theano.tensor as T

from .base import Layer

__all__ = [
    "LocalResponseNormalization2DLayer",
    "BatchNormLayer",
    "batch_norm",
]


class LocalResponseNormalization2DLayer(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    Input order is assumed to be `BC01`.

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from pylearn2. See the module docstring for license
    information.
    """

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """
        :parameters:
            - incoming: input layer or shape
            - alpha: see equation above
            - k: see equation above
            - beta: see equation above
            - n: number of adjacent channels to normalize over.
        """
        super(LocalResponseNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return input / scale


class BatchNormLayer(Layer):
    """
    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant added to the variance before taking the square root and
        dividing by it, to avoid numeric problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear (this is the default).
    mode : {'low_mem', 'high_mem'}
        Specify which batch normalization implementation to use: ``'low_mem'``
        avoids storing intermediate representations and thus requires less
        memory, while ``'high_mem'`` can reuse representations for the backward
        pass and is thus 5-10% faster.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1]: Ioffe, Sergey and Szegedy, Christian (2015):
            Batch Normalization: Accelerating Deep Network Training by Reducing
            Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes=None, epsilon=1e-4, alpha=0.5,
            nonlinearity=None, mode='low_mem', **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        self.mode = mode

        # create normalization values, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes)
            std = T.sqrt(input.var(self.axes) + self.epsilon)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(self.mean.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)
        beta = beta.dimshuffle(pattern)
        gamma = gamma.dimshuffle(pattern)

        # normalize
        # normalized = (input - mean) * (gamma / std) + beta
        normalized = T.nnet.batch_normalization(input, gamma=gamma, beta=beta,
                                                mean=mean, std=std,
                                                mode=self.mode)
        return self.nonlinearity(normalized)


def batch_norm(layer, mode='low_mem'):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    mode : {'low_mem', 'high_mem'}
        Specify which batch normalization implementation to use: ``'low_mem'``
        avoids storing intermediate representations and thus requires less
        memory, while ``'high_mem'`` can reuse representations for the backward
        pass and is thus 5-10% faster.

    Returns
    -------
    :class:`BatchNormLayer` instance
        A batch normalization layer stacked on the given modified `layer`.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity, mode=mode)
