from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jaxtyping import Key

from .data import BasinData

DAY_TO_S = 86400.0
KM2_TO_M2 = 1000000.0
KM_TO_M = 1000.0


class AbstractModel(eqx.Module):
    seq_length: int

    @abstractmethod
    def __call__(self, *args: Array):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_norms(
        data: BasinData,
    ) -> Tuple[
        Tuple[Array, Array],
        Tuple[Array, Array],
        Tuple[Array, Array],
    ]:
        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        data: BasinData,
        xd_norms: Tuple[Array, Array],
        xs_norms: Tuple[Array, Array],
        y_norms: Tuple[Array, Array],
    ):
        raise NotImplementedError

    @abstractmethod
    def serialize(
        self,
        data: BasinData,
    ) -> Tuple[Array, ...]:
        raise NotImplementedError


class AbstractConvolutionModel(AbstractModel):
    @abstractmethod
    def __call__(self, *args: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    def get_norms(
        data: BasinData,
    ) -> Tuple[
        Tuple[Array, Array],
        Tuple[Array, Array],
        Tuple[Array, Array],
    ]:
        # limamau: is this the best normalization?
        # maybe an alternative is to normalize all the attributes but the area
        # as usual, and then normalize the area by the std and divide the streamflow
        # by the same value.
        xd_norms = (
            jnp.zeros((data.xd.shape[-1],)),
            jnp.ones(
                (data.xd.shape[-1]),
            ),
        )
        xs_norms = (
            jnp.zeros((data.xs.shape[-1],)),
            jnp.ones(
                (data.xs.shape[-1]),
            ),
        )
        y_norms = jnp.array([0.0]), jnp.array([1.0])
        return xd_norms, xs_norms, y_norms

    def serialize(self, data: BasinData) -> Tuple[Array, Array, Array]:
        # pre-allocation to avoid possibly unbound check error
        xd_ser = jnp.array([])
        xs_ser = jnp.array([])
        y_ser = jnp.array([])

        flag = True
        for idx in range(data.y.shape[0]):
            xd, xs, y = data.get_single_basin(idx)
            num_sequences = xd.shape[0] - self.seq_length + 1
            xd_windows = jnp.stack(
                [xd[i : i + self.seq_length] for i in range(num_sequences)], axis=0
            )
            xs_windows = jnp.repeat(jnp.expand_dims(xs, axis=(0, -1)), num_sequences, 0)
            y_windows = y[self.seq_length - 1 :]

            if flag:
                xd_ser, xs_ser, y_ser = xd_windows, xs_windows, y_windows
                flag = False

            else:
                xd_ser = jnp.concatenate((xd_ser, xd_windows), axis=0)
                xs_ser = jnp.concatenate((xs_ser, xs_windows), axis=0)
                y_ser = jnp.concatenate((y_ser, y_windows), axis=0)

        y_ser = jnp.expand_dims(y_ser, axis=1)

        return xd_ser, xs_ser, y_ser

    def simulate(
        self,
        data: BasinData,
        xd_norms: Tuple[Array, Array],
        xs_norms: Tuple[Array, Array],
        y_norms: Tuple[Array, Array],
    ) -> Array:
        data.normalize(xd_norms, xs_norms, None)

        def scan_fn(carry, i):
            # extract seq_length slice
            xd_slice = jax.lax.dynamic_slice(
                data.xd,
                (0, i, 0),
                (data.xd.shape[0], self.seq_length, data.xd.shape[2]),
            )
            xs_expanded = jnp.expand_dims(data.xs, axis=(-1,))

            # predict single step
            y_pred = jax.vmap(self)(xd_slice, xs_expanded)
            y_pred = y_pred * y_norms[1] + y_norms[0]

            return carry, y_pred

        # predict through scan
        num_steps = data.y.shape[1] - self.seq_length + 1
        _, y_preds = jax.lax.scan(scan_fn, None, jnp.arange(num_steps))
        y_preds = jnp.transpose(y_preds)[0]

        return y_preds


class AbstractRecurrentModel(AbstractModel):
    @abstractmethod
    def __call__(self, *args: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    def get_norms(
        data: BasinData,
    ) -> Tuple[
        Tuple[Array, Array],
        Tuple[Array, Array],
        Tuple[Array, Array],
    ]:
        return data.get_norms()

    @staticmethod
    def _xs_to_xd(xs, xd):
        repeated_xs = jnp.repeat(jnp.expand_dims(xs, axis=-2), xd.shape[-2], axis=-2)
        return jnp.concatenate([xd, repeated_xs], axis=-1)

    def serialize(self, data: BasinData) -> Tuple[Array, Array]:
        # pre-allocations to avoid possibly unbound check error
        xd_ser = jnp.array([])
        y_ser = jnp.array([])

        flag = True
        for idx in range(data.y.shape[0]):
            xd, xs, y = data.get_single_basin(idx)
            xd = self._xs_to_xd(xs, xd)
            num_sequences = xd.shape[0] - self.seq_length + 1
            xd_windows = jnp.stack(
                [xd[i : i + self.seq_length] for i in range(num_sequences)], axis=0
            )
            y_windows = jnp.expand_dims(y[self.seq_length - 1 :], axis=1)

            if flag:
                xd_ser, y_ser = xd_windows, y_windows
                flag = False

            else:
                xd_ser = jnp.concatenate((xd_ser, xd_windows), axis=0)
                y_ser = jnp.concatenate((y_ser, y_windows), axis=0)

        return xd_ser, y_ser

    def simulate(
        self,
        data: BasinData,
        xd_norms: Tuple[Array, Array],
        xs_norms: Tuple[Array, Array],
        y_norms: Tuple[Array, Array],
    ) -> Array:
        data.normalize(xd_norms, xs_norms, None)
        xd = self._xs_to_xd(data.xs, data.xd)

        def scan_fn(carry, i):
            # extract seq_length slice
            xd_slice = jax.lax.dynamic_slice(
                xd, (0, i, 0), (xd.shape[0], self.seq_length, xd.shape[2])
            )

            # predict single step
            y_pred = jax.vmap(self)(xd_slice)
            y_pred = y_pred * y_norms[1] + y_norms[0]

            return carry, y_pred

        # predict through scan
        num_steps = data.y.shape[1] - self.seq_length + 1
        _, y_preds = jax.lax.scan(scan_fn, None, jnp.arange(num_steps))
        y_preds = jnp.transpose(y_preds)[0]

        return y_preds


class FixedGamma(AbstractConvolutionModel):
    shape: float
    scale: float
    is_conserving_mass: bool

    def __init__(
        self,
        shape: float,
        scale: float,
        seq_length: int,
        is_conserving_mass: bool = False,
    ):
        self.shape = shape
        self.scale = scale
        self.seq_length = seq_length
        self.is_conserving_mass = is_conserving_mass

    def __call__(
        self,
        *args: Array,
    ) -> Array:
        # get arguments
        xd, xs = args

        # areas
        areas = xs[..., 0, :]

        # gamma pdf
        pdf = jax.scipy.stats.gamma.pdf(
            x=jnp.arange(self.seq_length),
            a=self.seq_length,
            loc=self.shape,
            scale=self.scale,
        )

        if self.is_conserving_mass:
            pdf = pdf / jnp.sum(pdf)

        # runoff components
        sro_conv = jnp.sum(pdf[::-1] * xd[..., 0], axis=-1)
        ssro_conv = jnp.sum(pdf[::-1] * xd[..., 1], axis=-1)

        result = (sro_conv + ssro_conv) * areas / DAY_TO_S * KM2_TO_M2

        return result


class FixedIRF(AbstractConvolutionModel):
    C: float
    D: float
    is_conserving_mass: bool

    def __init__(
        self,
        velocity: float,
        diffusivity: float,
        seq_length: int,
        is_conserving_mass: bool = False,
    ):
        self.C = velocity
        self.D = diffusivity
        self.seq_length = seq_length
        self.is_conserving_mass = is_conserving_mass

    def __call__(
        self,
        *args: Array,
    ) -> Array:
        # get args
        xd, xs = args

        # distances
        distances = xs[..., -1, :]

        # time for the integral
        t = jnp.arange(1, self.seq_length + 1)

        # pdf (it's not really a pdf, but it's the same idea)
        pdf = (
            distances
            / (2 * t * jnp.sqrt(jnp.pi * self.D * t))
            * jnp.exp(-((self.C * t - distances) ** 2) / (4 * self.D * t))
        )

        if self.is_conserving_mass:
            pdf = pdf / jnp.sum(pdf)

        # runoff components
        up_q_conv = jnp.sum(pdf[::-1] * xd[..., -1], axis=-1)

        return up_q_conv


class LSTM(AbstractRecurrentModel):
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    hidden_size: int
    seq_length: int

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        seq_length: int,
        *,
        key: Key,
    ):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.seq_length = seq_length

    def __call__(
        self,
        *args: Array,
    ) -> Array:
        # get args
        xd = args[0]

        init_state = (
            jnp.zeros(self.hidden_size),  # h
            jnp.zeros(self.hidden_size),  # c
        )

        def scan_fn(state, input):
            return (self.cell(input, state), None)

        (h, _), _ = jax.lax.scan(scan_fn, init_state, xd)
        return self.linear(h)


class MLPGamma(AbstractConvolutionModel):
    hidden_size: int
    attributes_size: int
    multiplier: float
    is_conserving_mass: bool
    slinear1: eqx.nn.Linear
    sslinear1: eqx.nn.Linear
    slinear2: eqx.nn.Linear
    sslinear2: eqx.nn.Linear

    def __init__(
        self,
        attributes_size: int,
        hidden_size: int,
        seq_length: int,
        multiplier: float,
        key: Key,
        is_conserving_mass: bool = False,
    ):
        skey1, sskey1, skey2, sskey2 = jrandom.split(key, num=4)
        self.attributes_size = attributes_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.multiplier = multiplier
        self.slinear1 = eqx.nn.Linear(attributes_size, hidden_size, key=skey1)
        self.sslinear1 = eqx.nn.Linear(attributes_size, hidden_size, key=sskey1)
        self.slinear2 = eqx.nn.Linear(hidden_size, 2, key=skey2)
        self.sslinear2 = eqx.nn.Linear(hidden_size, 2, key=sskey2)
        self.is_conserving_mass = is_conserving_mass

    # limamau: why is this forecasting NaNs? more attributes is enough?
    # limamau: use sro and ssro instead of xd
    # limamau: use areas and other attributes instead of xs
    def __call__(
        self,
        *args: Array,
    ) -> Array:
        # get args
        xd, xs = args

        # MLP
        s_hidden = jax.nn.relu(jax.vmap(self.slinear1)(xs))
        ss_hidden = jax.nn.relu(jax.vmap(self.sslinear1)(xs))
        s_params = jax.nn.sigmoid(jax.vmap(self.slinear2)(s_hidden)) * self.multiplier
        ss_params = (
            jax.nn.sigmoid(jax.vmap(self.sslinear2)(ss_hidden)) * self.multiplier
        )

        # for consistency
        s_params = jnp.clip(s_params, 1e-5, self.multiplier)
        ss_params = jnp.clip(ss_params, 1e-5, self.multiplier)

        # get params
        s_shapes, s_scales = s_params[:, 0], s_params[:, 1]
        ss_shapes, ss_scales = ss_params[:, 0], ss_params[:, 1]

        # distributions (from MLP)
        s_pdf = jax.scipy.stats.gamma.pdf(
            x=jnp.arange(self.seq_length),
            a=s_shapes,
            scale=s_scales,
        )
        ss_pdf = jax.scipy.stats.gamma.pdf(
            x=jnp.arange(self.seq_length),
            a=ss_shapes,
            scale=ss_scales,
        )

        if self.is_conserving_mass:
            s_pdf = s_pdf / jnp.sum(s_pdf)
            ss_pdf = ss_pdf / jnp.sum(ss_pdf)

        # runoff components
        sro_conv = jnp.sum(s_pdf[::-1] * xd[..., 0], axis=-1)
        ssro_conv = jnp.sum(ss_pdf[::-1] * xd[..., 1], axis=-1)

        return (sro_conv + ssro_conv) * xs[:, 0] / DAY_TO_S * KM2_TO_M2
