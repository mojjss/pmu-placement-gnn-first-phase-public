from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def fake_net():
    """Small pandapower-like object with parallel lines and one transformer."""
    return SimpleNamespace(
        bus=pd.DataFrame(
            {"vn_kv": [110.0, 110.0, 20.0, 10.0], "in_service": [True] * 4},
            index=[0, 1, 2, 3],
        ),
        line=pd.DataFrame(
            {
                "from_bus": [0, 0, 1],
                "to_bus": [1, 1, 2],
                "length_km": [10.0, 12.0, 5.0],
                "r_ohm_per_km": [0.1, 0.2, 0.3],
                "in_service": [True, True, True],
            },
            index=[0, 1, 2],
        ),
        trafo=pd.DataFrame(
            {
                "hv_bus": [2],
                "lv_bus": [3],
                "sn_mva": [25.0],
                "vk_percent": [10.0],
                "in_service": [True],
            },
            index=[0],
        ),
        trafo3w=pd.DataFrame(
            columns=[
                "hv_bus",
                "mv_bus",
                "lv_bus",
                "sn_hv_mva",
                "sn_mv_mva",
                "sn_lv_mva",
                "vk_hv_percent",
                "vk_mv_percent",
                "vk_lv_percent",
                "in_service",
            ]
        ),
        load=pd.DataFrame({"bus": [1], "in_service": [True]}),
        gen=pd.DataFrame({"bus": [2], "in_service": [True]}),
        sgen=pd.DataFrame(columns=["bus", "in_service"]),
        ext_grid=pd.DataFrame({"bus": [0], "in_service": [True]}),
    )

