"""
DISCLAIMER: This code originates from pypsa-eur
(see: https://github.com/PyPSA/pypsa-eur/blob/master/scripts/prepare_network.py)
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pypsa

import logging

logger = logging.getLogger(__name__)


def average_every_nhours(net: pypsa.Network, offset: str, precision: int = 3) -> pypsa.Network:
    """
    Average all components snapshots to a given offset and update network snapshot weights

    Parameters
    ----------
    net: pypsa.Network
        PyPSA network
    offset: str
        Offset over which the mean is applied (e.g. 3H, 1D, etc.)
    precision: int (default: 3)
        Indicates at which decimal time series should be rounded
    """
    logger.info(f"Resampling the network to {offset}")
    net_agg = net.copy(with_time=False)

    snapshot_weightings = net.snapshot_weightings.resample(offset).sum()
    net_agg.set_snapshots(snapshot_weightings.index)
    net_agg.snapshot_weightings = snapshot_weightings

    for c in net.iterate_components():
        pnl = getattr(net_agg, c.list_name+"_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean().round(precision)

    return net_agg


def apply_time_segmentation(n, segments):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError("Optional dependency 'tsam' not found."
                                  "Install via 'pip install tsam'")

    p_max_pu_norm = n.generators_t.p_max_pu.max()
    p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

    load_norm = n.loads_t.p_set.max()
    load = n.loads_t.p_set / load_norm

    inflow_norm = n.storage_units_t.inflow.max()
    inflow = n.storage_units_t.inflow / inflow_norm

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(raw, hoursPerPeriod=len(raw),
                                     noTypicalPeriods=1, noSegments=int(segments),
                                     segmentation=True)

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name='name'))
    n.snapshot_weightings['objective'] = pd.Series(weightings, index=snapshots, name="objective", dtype="float64")
    n.snapshot_weightings['generators'] = pd.Series(weightings, index=snapshots, name="generator", dtype="float64")
    n.snapshot_weightings['stores'] = pd.Series(n.config['time']['resolution'], index=snapshots, name="store", dtype="float64")

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n


def apply_time_reduction(n, no_periods, no_hours_per_period):

    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError("Optional dependency 'tsam' not found."
                                  "Install via 'pip install tsam'")

    p_max_pu = n.generators_t.p_max_pu
    load = n.loads_t.p_set
    inflow = n.storage_units_t.inflow

    dataset = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(dataset,
                                     hoursPerPeriod=no_hours_per_period,
                                     noTypicalPeriods=no_periods,
                                     rescaleClusterPeriods=False,
                                     clusterMethod='hierarchical')
    reduced = agg.createTypicalPeriods()

    dataset['indexint'] = range(len(dataset))
    dataset['time'] = dataset['indexint'] // no_hours_per_period

    indices = []
    weights = []
    for idx, center in enumerate(agg.clusterCenterIndices):
        indices.extend(list(dataset.index[dataset.time == center]))
        weights.extend([agg.clusterPeriodNoOccur[idx]]*no_hours_per_period)

    n.set_snapshots(sorted(indices))
    n.snapshot_weightings['objective'] = pd.Series(weights, index=indices,
                                                   name="objective", dtype="float64")
    n.snapshot_weightings['generators'] = pd.Series(weights, index=indices,
                                                    name="generator", dtype="float64")
    n.snapshot_weightings['stores'] = pd.Series(n.config['time']['resolution'], index=indices,
                                                name="store", dtype="float64")

    reduced.index = indices
    reduced = reduced.sort_index()

    n.snapshot_weightings = n.snapshot_weightings.sort_index()

    n.generators_t.p_max_pu = reduced[n.generators_t.p_max_pu.columns]
    n.loads_t.p_set = reduced[n.loads_t.p_set.columns]
    n.storage_units_t.inflow = reduced[n.storage_units_t.inflow.columns]

    return n
