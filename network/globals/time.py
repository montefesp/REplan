"""
DISCLAIMER: This code originates from pypsa-eur
(see: https://github.com/PyPSA/pypsa-eur/blob/master/scripts/prepare_network.py)
"""
import numpy as np
import pandas as pd
import pypsa

import logging

logger = logging.getLogger(__name__)


def average_every_nhours(net: pypsa.Network, offset: str) -> pypsa.Network:
    """
    Average all components snapshots to a given offset and update network snapshot weights

    Parameters
    ----------
    net: pypsa.Network
        PyPSA network
    offset: str
        Offset over which the mean is applied (e.g. 3H, 1D, etc.)
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
                pnl[k] = df.resample(offset).mean()

    return net_agg


def apply_time_segmentation(n, segments, solver_name):
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

    # solver_name = snakemake.config["solving"]["solver"]["name"]

    agg = tsam.TimeSeriesAggregation(raw, hoursPerPeriod=len(raw),
                                     noTypicalPeriods=1, noSegments=int(segments),
                                     segmentation=True, solver=solver_name)

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name='name'))
    n.snapshot_weightings = pd.Series(weightings, index=snapshots, name="weightings", dtype="float64")

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n
