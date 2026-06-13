"""JAX track data and Frenet helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gymkhana.envs.track import Track


class JaxTrack(NamedTuple):
    xs: jnp.ndarray
    ys: jnp.ndarray
    ss: jnp.ndarray
    yaws: jnp.ndarray
    curvatures: jnp.ndarray
    widths: jnp.ndarray
    length: float

    @classmethod
    def from_track_name(cls, name: str, track_scale: float = 1.0, reversed: bool = False) -> "JaxTrack":
        track = Track.from_track_name(name, track_scale=track_scale)
        track.set_direction(reversed)
        return cls.from_track(track)

    @classmethod
    def from_track(cls, track: Track) -> "JaxTrack":
        centerline = track.centerline
        if centerline is None:
            raise ValueError("JaxTrack requires a track with centerline data.")
        if centerline.ss is None or centerline.w_lefts is None or centerline.w_rights is None:
            raise ValueError("JaxTrack requires centerline arc length and width data.")

        length = float(centerline.spline.s[-1])
        widths = np.asarray(centerline.w_lefts + centerline.w_rights, dtype=np.float32)
        return cls(
            xs=jnp.asarray(centerline.xs, dtype=jnp.float32),
            ys=jnp.asarray(centerline.ys, dtype=jnp.float32),
            ss=jnp.asarray(centerline.ss, dtype=jnp.float32),
            yaws=jnp.asarray(centerline.yaws, dtype=jnp.float32),
            curvatures=jnp.asarray(centerline.ks, dtype=jnp.float32),
            widths=jnp.asarray(widths, dtype=jnp.float32),
            length=length,
        )

    @property
    def closed_xs(self):
        return jnp.concatenate([self.xs, self.xs[:1]])

    @property
    def closed_ys(self):
        return jnp.concatenate([self.ys, self.ys[:1]])

    @property
    def closed_ss(self):
        return jnp.concatenate([self.ss, jnp.asarray([self.length], dtype=self.ss.dtype)])


def wrap_angle(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def project_to_centerline(track: JaxTrack, x, y, yaw):
    """Project batched Cartesian poses to approximate Frenet coordinates."""
    pts = jnp.stack([track.closed_xs, track.closed_ys], axis=-1)
    seg_start = pts[:-1]
    seg_end = pts[1:]
    diffs = seg_end - seg_start
    l2s = jnp.sum(diffs * diffs, axis=-1)

    point = jnp.stack([x, y], axis=-1)
    point_to_start = point[..., None, :] - seg_start
    dots = jnp.sum(point_to_start * diffs, axis=-1)
    t = jnp.clip(dots / l2s, 0.0, 1.0)
    projections = seg_start + t[..., None] * diffs
    dist_vec = point[..., None, :] - projections
    dist2 = jnp.sum(dist_vec * dist_vec, axis=-1)
    idx = jnp.argmin(dist2, axis=-1)

    t_best = jnp.take_along_axis(t, idx[..., None], axis=-1)[..., 0]
    dist_best = jnp.sqrt(jnp.take_along_axis(dist2, idx[..., None], axis=-1)[..., 0])
    s_start = track.closed_ss[:-1]
    s_end = track.closed_ss[1:]
    s = s_start[idx] + t_best * (s_end[idx] - s_start[idx])
    s = jnp.mod(s, track.length)

    yaw_track = track.yaws[idx % track.yaws.shape[0]]
    normal = jnp.stack([-jnp.sin(yaw_track), jnp.cos(yaw_track)], axis=-1)
    proj_best = jnp.take_along_axis(projections, idx[..., None, None], axis=-2)[..., 0, :]
    sign = jnp.sign(jnp.sum((point - proj_best) * normal, axis=-1))
    ey = dist_best * sign
    ephi = wrap_angle(yaw - yaw_track)
    return s, ey, ephi, idx


def nearest_by_s(values, ss, query_s):
    s_wrapped = jnp.mod(query_s, ss[-1])
    idx = jnp.argmin(jnp.abs(ss - s_wrapped[..., None]), axis=-1)
    return values[idx]


def sample_lookahead(track: JaxTrack, s, n_points: int = 5, ds: float = 0.5, sparse_widths: bool = True):
    offsets = jnp.arange(1, n_points + 1, dtype=s.dtype) * ds
    s_ahead = s[..., None] + offsets
    curvatures = nearest_by_s(track.curvatures, track.ss, s_ahead)
    widths = nearest_by_s(track.widths, track.ss, s_ahead)
    if sparse_widths:
        widths = jnp.stack([widths[..., 0], widths[..., -1]], axis=-1)
    return curvatures, widths
