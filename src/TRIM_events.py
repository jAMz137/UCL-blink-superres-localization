"""Blinking event detection from frame-difference image stacks."""

from __future__ import annotations

import numpy as np

from .TRIM_config import AnalysisConfig
from .utils_toolbox import imloc_max, sci_opt_fit


class BlinkEvent:
    """A connected spatiotemporal blinking-transition candidate."""

    def __init__(self, slc: tuple[np.ndarray, np.ndarray]):
        self.min_coord, self.max_coord = slc
        self.zng = (self.max_coord[0] + 1 - self.min_coord[0]) / 2
        self.glced = False
        self.brk_mrk = 0
        self.std_err = 10.0

    @property
    def centr(self) -> np.ndarray:
        return self._centr

    @centr.setter
    def centr(self, ary: np.ndarray) -> None:
        if np.any(ary < 0):
            raise ValueError("Event center indices must be non-negative.")
        tz = (self.max_coord[0] + 1 + self.min_coord[0]) / 2
        self._centr = np.append(tz, ary)

    @property
    def shapeR(self) -> np.ndarray:
        """Spatial y/x width of the event region."""

        return np.array(
            [
                self.max_coord[1] - self.min_coord[1],
                self.max_coord[2] - self.min_coord[2],
            ]
        )


class BlinkEventSet:
    """Detect, validate, and pair blinking events.

    Input arrays are the raw and Gaussian-smoothed frame-difference stacks.
    Output lists contain positive and negative events plus a boolean glitch mask.
    """

    def __init__(
        self,
        image_diff: np.ndarray,
        gaussian_image_diff: np.ndarray,
        config: AnalysisConfig,
    ):
        self.imdi = image_diff
        self.imgd = gaussian_image_diff
        self.shape1 = np.shape(gaussian_image_diff)
        self.config = config
        event_extent = int(2 * config.spot_sigma_px)
        slice_p, posp = imloc_max(
            gaussian_image_diff,
            3.5,
            9.0,
            (3, event_extent, event_extent),
            event_extent,
        )
        slice_n, posn = imloc_max(
            gaussian_image_diff,
            -3.5,
            9.0,
            (3, event_extent, event_extent),
            event_extent,
        )

        self.events_p = self.mark_event(slice_p, posp)
        self.events_n = self.mark_event(slice_n, posn)
        self.glitch_id = self.pair_glitch()

    def mark_event(
        self,
        slices: list[tuple[np.ndarray, np.ndarray]],
        max_positions: list[np.ndarray],
    ) -> list[BlinkEvent]:
        """Fit each candidate event and reject malformed candidates."""

        cfg = self.config
        event_extent = int(2 * cfg.spot_sigma_px)
        edge_padding = int(np.ceil(cfg.spot_sigma_px / 2) + 1)
        events = []
        for j, slc in enumerate(slices):
            event = BlinkEvent(slc)
            shape_r = event.shapeR
            break_mark = event.brk_mrk

            spot_expanded = np.abs(
                np.sum(
                    self.imdi[
                        event.min_coord[0] : event.max_coord[0] + 1,
                        max(event.min_coord[1] - edge_padding, 0) : event.max_coord[1]
                        + edge_padding
                        + 1,
                        max(event.min_coord[2] - edge_padding, 0) : event.max_coord[2]
                        + edge_padding
                        + 1,
                    ],
                    axis=0,
                )
            )
            spot_core = np.abs(
                np.sum(
                    self.imgd[
                        event.min_coord[0] : event.max_coord[0] + 1,
                        event.min_coord[1] : event.max_coord[1] + 1,
                        event.min_coord[2] : event.max_coord[2] + 1,
                    ],
                    axis=0,
                )
            )
            peak_core = np.argwhere(spot_core == np.max(spot_core))[0]
            center_from_core = peak_core + np.array(
                [event.min_coord[1], event.min_coord[2]]
            )
            center_from_filter = max_positions[j][1:]
            peak_filter = center_from_filter - np.array(
                [event.min_coord[1], event.min_coord[2]]
            )

            if np.any(np.abs(peak_filter - peak_core) > event_extent / 2):
                center_from_core = center_from_filter
                peak_core = peak_filter
                break_mark = 1
            if np.any(np.abs(peak_core + 1 - shape_r / 2) > event_extent):
                break_mark = 1
            if np.any(shape_r < event_extent):
                break_mark = 3
            elif not (
                center_from_core[0] - event_extent >= 0
                and center_from_core[1] - event_extent >= 0
                and center_from_core[0] + event_extent <= self.shape1[1] - 1
                and center_from_core[1] + event_extent <= self.shape1[2] - 1
            ):
                break_mark = 4

            if break_mark <= 1:
                fit = sci_opt_fit(spot_expanded, cfg.pixel_size, 4.6)
                popt, status, perr = fit
                if status == "fail":
                    break_mark = 2
                else:
                    event.std_err = float(sum(perr[1:3]))
                    if event.std_err < 1:
                        center_from_core = np.array(
                            [
                                popt[2] + max(event.min_coord[1] - edge_padding, 0),
                                popt[1] + max(event.min_coord[2] - edge_padding, 0),
                            ]
                        )

            event.centr = center_from_core
            event.brk_mrk = break_mark
            events.append(event)
        return events

    def __getitem__(self, idx: int) -> list[BlinkEvent]:
        if idx == 0:
            return self.events_p
        if idx == 1:
            return self.events_n
        raise IndexError(f"index {idx} is out of range")

    def pairing(self, first: BlinkEvent, second: BlinkEvent):
        """Pair nearby same-polarity events that likely represent a glitch."""

        center_z0 = np.int32(first.centr[0])
        center_z1 = np.int32(second.centr[0])
        z0 = np.floor(center_z0 - first.zng - second.zng - 2)
        z1 = np.ceil(center_z0 + first.zng + second.zng + 2 + 1)

        if center_z1 not in range(np.int32(z0), np.int32(z1)):
            return []

        first_min = first.min_coord
        first_max = first.max_coord
        second_min = second.min_coord
        second_max = second.max_coord
        lower = np.min(np.vstack((first_min, second_min)), axis=0)
        upper = np.max(np.vstack((first_max, second_max)), axis=0)
        spot_first = np.sum(
            self.imdi[
                first_min[0] : first_max[0] + 1,
                lower[1] : upper[1] + 1,
                lower[2] : upper[2] + 1,
            ],
            axis=0,
        )
        spot_second = np.sum(
            self.imdi[
                second_min[0] : second_max[0] + 1,
                lower[1] : upper[1] + 1,
                lower[2] : upper[2] + 1,
            ],
            axis=0,
        )
        diff = np.abs(np.mean(spot_first + spot_second))
        slc = [
            slice(lower[0] + 1, upper[0] + 1),
            slice(lower[1], upper[1] + 1),
            slice(lower[2], upper[2] + 1),
        ]
        if diff < 3.5:
            return diff, slc
        return []

    def pair_glitch(self) -> np.ndarray:
        """Create a boolean mask for paired same-polarity glitch regions."""

        negative_events = [item for item in self.events_n if item.brk_mrk <= 2]
        positive_events = [item for item in self.events_p if item.brk_mrk <= 2]
        if not negative_events:
            return np.full((self.shape1[0] + 1, self.shape1[1], self.shape1[2]), False)

        centers_n = np.array([item.centr for item in negative_events])
        shape2 = (self.shape1[0] + 1, self.shape1[1], self.shape1[2])
        marked_id = np.full(shape2, False, dtype=bool)

        for item00 in positive_events:
            if item00.brk_mrk == 1:
                continue
            center_z, center_y, center_x = np.array(np.int32(item00.centr))
            event_extent = int(2 * self.config.spot_sigma_px)
            locidxy = np.where(
                (centers_n[:, 0] > center_z - 5)
                & (centers_n[:, 0] < center_z + 5)
                & (centers_n[:, 1] >= np.floor(center_y - (event_extent + 1)))
                & (centers_n[:, 1] <= np.ceil(center_y + (event_extent + 1)))
                & (centers_n[:, 2] >= np.floor(center_x - (event_extent + 1)))
                & (centers_n[:, 2] <= np.ceil(center_x + (event_extent + 1)))
            )
            if len(locidxy[0]) == 0:
                continue
            z_value = np.array(np.int32(centers_n[locidxy, 0])).T
            pairn = 0

            locidgrt = np.where(z_value >= center_z)[0]
            if len(locidgrt) != 0:
                locid1 = np.argmin(z_value[locidgrt])
                locidz1 = locidxy[0][locidgrt[locid1]]
                item11 = negative_events[locidz1]
                if (not item11.glced) and item11.brk_mrk != 1:
                    try:
                        diff1, slc1 = self.pairing(item00, item11)
                        pairn += 1
                    except (TypeError, ValueError):
                        pass

            locidles = np.where(z_value <= center_z)[0]
            if len(locidles) != 0:
                locid2 = np.argmax(z_value[locidles])
                locidz2 = locidxy[0][locidles[locid2]]
                item12 = negative_events[locidz2]
                if (not item12.glced) and item12.brk_mrk != 1:
                    try:
                        diff2, slc2 = self.pairing(item00, item12)
                        pairn += 2
                    except (TypeError, ValueError):
                        pass

            if pairn == 1 or (pairn == 3 and diff1 <= diff2):  # type: ignore[name-defined]
                item00.glced = True
                item11.glced = True  # type: ignore[name-defined]
                marked_id[slc1[0], slc1[1], slc1[2]] = True  # type: ignore[name-defined]
            elif pairn == 2 or (pairn == 3 and diff1 > diff2):  # type: ignore[name-defined]
                item00.glced = True
                item12.glced = True  # type: ignore[name-defined]
                marked_id[slc2[0], slc2[1], slc2[2]] = True  # type: ignore[name-defined]
        return marked_id


blk_event = BlinkEvent
blk_events = BlinkEventSet
