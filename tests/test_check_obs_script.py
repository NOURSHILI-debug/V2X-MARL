import numpy as np

from scripts.check_obs import (
    BLOCKS,
    OBS_DIM,
    _new_saturation_counters,
    _validate_observations,
)


def test_fake_observations_initialize_and_update_saturation_for_each_block():
    counters = _new_saturation_counters()
    stats = {name: [float("inf"), float("-inf")] for name in BLOCKS}
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0] = 1.0
    observation[-1] = 1.0

    _validate_observations({"agent_0": observation}, ["agent_0"], stats, counters)

    assert all(name in counters for name in BLOCKS)
    assert counters["ego"] == 1
    assert counters["ego_total"] == BLOCKS["ego"].stop - BLOCKS["ego"].start
    assert counters["mask"] == 1
    assert counters["mask_total"] == BLOCKS["mask"].stop - BLOCKS["mask"].start
    assert counters["ego_0"] == 1
    assert counters["mask_5"] == 1
