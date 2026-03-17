import numpy as np
import pandas as pd


def mutant_names_from_F(F):
    """
    Extract mutant names from path strings like 'WT/Simulation_1'.
    """
    return np.array([p.split("/")[0] for p in F], dtype=object)


def occupancy_from_frame_labels(frame_mutant, labels):
    """
    Compute occupancy table from frame-level mutant names and frame-level labels.
    """
    df = pd.DataFrame({
        "mutant": frame_mutant,
        "state": labels
    })

    counts = (
        df.groupby(["mutant", "state"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )

    fractions = counts.div(counts.sum(axis=1), axis=0)
    return counts, fractions


def occupancy_from_traj_labels(F, nframes_sub, labels):
    """
    For flattened labels built from trajectory order x frame order.
    """
    traj_mutant = mutant_names_from_F(F)
    frame_mutant = np.repeat(traj_mutant, nframes_sub)
    return occupancy_from_frame_labels(frame_mutant, labels)
