"""Non-blinking interval extraction and differential spot construction."""

from __future__ import annotations

import numpy as np

from .TRIM_config import AnalysisConfig
from .TRIM_events import BlinkEvent, BlinkEventSet
from .utils_toolbox import consec_T, consec_T3, gen_circle


class NonBlinkingInterval:
    """Stable frames before and after one blinking event.

    The interval stores the local time trace, accepted frame masks, the
    differential single-particle spot, and later the localization result.
    """

    def __init__(
        self,
        excitation: float,
        glitch_or_not: bool,
        center: np.ndarray,
        time_marks: list[int],
    ):
        self.abort = 0
        self.glitch_or_not = glitch_or_not
        self.excitation = excitation
        self.rng_t0 = time_marks
        self.cnt_12 = center
        self.Imint: np.ndarray | None = None
        self.Imspt: np.ndarray | None = None
        self.ValidC = 0.0
        self.SpotA: np.ndarray | None = None
        self.SpotI = self.cnt_z
        self.SpotB: np.ndarray | None = None
        self.corner: list[int] = [0, 0]
        self.xy: list[float] = [0.0, 0.0]
        self.drift: list[float] = [0.0, 0.0]
        self.fit_o: dict = {}
        self.weight = 0

    @property
    def cnt_z(self) -> float:
        return (self.rng_t0[3] + self.rng_t0[0]) / 2

    @property
    def Dint(self) -> float:
        if self.Imint is None:
            return 0.0
        return float(
            np.abs(
                self.Imint[self.rng_t0[1] - self.rng_t0[0]]
                - self.Imint[self.rng_t0[2] - self.rng_t0[0]]
            )
        )

    @property
    def frame_wght(self) -> int:
        return self.weight

    @frame_wght.setter
    def frame_wght(self, valid_frames: np.ndarray) -> None:
        self.weight = int(np.count_nonzero(valid_frames[0]) + np.count_nonzero(valid_frames[1]))

    @property
    def x(self) -> float:
        return self.xy[0]

    @property
    def y(self) -> float:
        return self.xy[1]


class NonBlinkingIntervalSet:
    """Build non-blinking intervals and isolate single-particle spots."""

    def __init__(
        self,
        events: BlinkEventSet,
        image_stack: np.ndarray,
        config: AnalysisConfig,
        excitation: float,
    ):
        self.glitch_id = events.glitch_id
        self.config = config
        self.excitation = excitation
        self.im = image_stack
        self.shape1 = np.shape(image_stack)
        eventp_all = [item for item in events[0] if item.brk_mrk <= 2]
        eventn_all = [item for item in events[1] if item.brk_mrk <= 2]
        eventp_t = [item for item in eventp_all if not item.glced]
        eventn_t = [item for item in eventn_all if not item.glced]
        centers = [item.centr for item in eventp_t + eventn_t]
        if centers:
            self.centrAll = np.array(centers)
            self.EventpnT = eventp_t + eventn_t
        else:
            self.centrAll = np.empty((0, 3))
            self.EventpnT = []

        self.Traces = self.comb_events(eventp_all) + self.comb_events(eventn_t)
        self.time_trace()
        valid_dints = np.array([item.Dint for item in self.Traces if item.Imint is not None])
        if valid_dints.size == 0:
            self.int_dif = 0.0
        else:
            keep = np.abs(valid_dints - np.mean(valid_dints)) < 2 * np.std(valid_dints)
            self.int_dif = float(np.mean(valid_dints[keep])) if np.any(keep) else float(np.mean(valid_dints))
        self.find_spot()

    def comb_events(self, events: list[BlinkEvent]) -> list[NonBlinkingInterval]:
        cfg = self.config
        event_extent = int(2 * cfg.spot_sigma_px)
        overlap_distance = event_extent * 0.8
        traces = []
        if self.centrAll.size == 0:
            return traces

        for event in events:
            if event.brk_mrk == 1:
                continue
            center_z, center_y, center_x = np.array(np.int32(event.centr))
            loc = np.where(
                (self.centrAll[:, 2] >= np.int32(center_x - overlap_distance))
                & (self.centrAll[:, 2] < np.int32(center_x + overlap_distance) + 1)
                & (self.centrAll[:, 1] >= np.int32(center_y - overlap_distance))
                & (self.centrAll[:, 1] < np.int32(center_y + overlap_distance) + 1)
                & (self.centrAll[:, 0] >= np.int32(center_z - 30))
                & (self.centrAll[:, 0] < np.int32(center_z + 30))
            )
            candidate_z = np.array(np.int32(self.centrAll[loc, 0])).T
            candidate_ids = loc[0]

            greater = np.where(candidate_z > center_z)[0]
            if len(greater):
                d4 = self.EventpnT[candidate_ids[greater[np.argmin(candidate_z[greater])]]].min_coord[0]
            else:
                d4 = min(center_z + 30, self.shape1[0] - 1)

            lesser = np.where(candidate_z < center_z)[0]
            if len(lesser):
                d1 = self.EventpnT[candidate_ids[lesser[np.argmax(candidate_z[lesser])]]].max_coord[0] + 1
            else:
                d1 = max(center_z - 30, 0)

            d2 = event.min_coord[0]
            d3 = event.max_coord[0] + 1
            if d2 - d1 <= 3 or d4 - d3 <= 3:
                continue
            traces.append(
                NonBlinkingInterval(
                    self.excitation,
                    event.glced,
                    event.centr[1:],
                    [int(d1), int(d2), int(d3), int(d4)],
                )
            )
        return traces

    def time_trace(self) -> None:
        """Extract local stacks and integrated time traces for each interval."""

        cfg = self.config
        interval_radius = int(2 * cfg.spot_sigma_px) + int(np.ceil(cfg.spot_sigma_px / 2) + 1)
        for interval in self.Traces:
            d1, _, _, d4 = interval.rng_t0
            mask = np.zeros(self.shape1[1:])
            x_circle, y_circle = gen_circle(interval.cnt_12, interval_radius, self.shape1[1:])
            mask[y_circle, x_circle] = 1
            inner = np.where(mask == 1)

            visited = np.zeros(self.shape1[1:], dtype=bool)
            for j in range(len(inner[0])):
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        yind = inner[0][j] + y_offset
                        xind = inner[1][j] + x_offset
                        if (
                            0 <= xind < self.shape1[2]
                            and 0 <= yind < self.shape1[1]
                            and mask[yind, xind] == 0
                            and not visited[yind, xind]
                        ):
                            mask[yind, xind] = 2
                            visited[yind, xind] = True

            min_y, min_x = np.array(inner).min(axis=1)
            max_y, max_x = np.array(inner).max(axis=1)
            local_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
            interval.Imspt = self.im[d1 : d4 + 1, min_y : max_y + 1, min_x : max_x + 1].copy()
            masked_stack = interval.Imspt.copy()

            inactive_pixels = np.array(np.where((local_mask == 2) | (local_mask == 0))).T
            marked = self.glitch_id[d1 : d4 + 1, min_y : max_y + 1, min_x : max_x + 1]
            invalid_frames = np.ones(len(masked_stack), dtype=bool)
            for k in range(len(masked_stack)):
                glitch_fraction = np.count_nonzero(marked[k]) / marked[k].size * 100
                if glitch_fraction >= 10:
                    invalid_frames[k] = False
                masked_stack[k, inactive_pixels[:, 0], inactive_pixels[:, 1]] = 95
            masked_stack[masked_stack < 95] = 95
            interval.invalid_frames = invalid_frames
            interval.Imint = np.mean(masked_stack, axis=(1, 2))
            interval.corner = [int(min_y), int(min_x)]

    @staticmethod
    def brk_ix(xi: np.ndarray, dd: int) -> np.ndarray:
        flag = 0
        for i, _ in enumerate(xi[dd:]):
            if flag:
                xi[i + dd] = False
            elif i > 15 and not bool(xi[i + dd - 1]):
                xi[i + dd] = False
                flag = 1
        return xi

    def find_spot(self) -> None:
        """Average stable frames on both sides and subtract them."""

        if self.int_dif <= 0:
            for interval in self.Traces:
                interval.abort = 1
            return

        for interval in self.Traces:
            if interval.Imint is None or interval.Imspt is None:
                interval.abort = 2
                continue
            if interval.Dint > 2 * self.int_dif or interval.Dint < 0.3 * self.int_dif:
                interval.abort = 1
                continue

            d1, d2, d3, _ = interval.rng_t0
            imint = interval.Imint
            int0 = imint[d2 - d1]
            int1 = imint[d3 - d1]
            midint = (int0 + int1) / 2
            length = len(imint)
            std_window = self.int_dif / 5
            interval.SpotA = np.mean(interval.Imspt[imint >= midint], axis=0) - np.mean(
                interval.Imspt[imint < midint], axis=0
            )
            interval.SpotI = (interval.rng_t0[3] + interval.rng_t0[0]) / 2

            int00 = int0
            int11 = int1
            for _ in range(11):
                idx = np.arange(length)
                x0 = (imint >= int00 - std_window) & (imint <= int00 + std_window) & (idx <= d2 - d1)
                x1 = (imint >= int11 - std_window) & (imint <= int11 + std_window) & (idx >= d3 - d1)
                if not (np.any(x0) and np.any(x1)):
                    interval.abort = 2
                    break
                int00 = (np.max(imint[x0]) + np.min(imint[x0])) / 2
                int11 = (np.max(imint[x1]) + np.min(imint[x1])) / 2
            if interval.abort:
                continue

            if np.abs(int0 - int00) >= std_window:
                int00 = int0
            if np.abs(int1 - int11) >= std_window:
                int11 = int1

            x0 = (imint >= int00 - std_window) & (imint <= int00 + std_window)
            x1 = (imint >= int11 - std_window) & (imint <= int11 + std_window)
            x00 = (imint >= int00 - 2 * std_window) & (imint <= int00 + 2 * std_window)
            x11 = (imint >= int11 - 2 * std_window) & (imint <= int11 + 2 * std_window)
            x0 = consec_T(x0, 3)
            x0 = consec_T3(x0, x00, 3)
            x1 = consec_T(x1, 3)
            x1 = consec_T3(x1, x11, 3)
            x0 = self.brk_ix(x0[::-1], len(x0) - d2 + d1)[::-1]
            x1 = self.brk_ix(x1, d3 - d1)
            if not (np.any(x0) and np.any(x1)):
                interval.abort = 3
                continue

            state0 = np.mean(interval.Imspt[x0], axis=0)
            state1 = np.mean(interval.Imspt[x1], axis=0)
            valid_counts = np.count_nonzero(x0), np.count_nonzero(x1)
            interval.ValidC = float(
                np.abs(
                    np.sum(imint[x0] - 95) * valid_counts[1]
                    - np.sum(imint[x1] - 95) * valid_counts[0]
                )
            )
            interval.SpotB = state0 - state1 if int0 > int1 else state1 - state0
            interval.frame_wght = np.array([x0, x1])


blk_trace = NonBlinkingInterval
blk_traces = NonBlinkingIntervalSet
