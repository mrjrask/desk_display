import importlib

travel_map = importlib.import_module("screens.draw_travel_map")


def test_extract_polylines_falls_back_to_step_polylines():
    route = {
        "legs": [
            {
                "steps": [
                    {
                        "polyline": {
                            "points": "_p~iF~ps|U_ulLnnqC_mqNvxq`@",
                        }
                    }
                ]
            }
        ]
    }

    polylines = travel_map._extract_polylines({"lake_shore": route})

    assert "lake_shore" in polylines
    # The sample polyline decodes into three coordinate points.
    assert len(polylines["lake_shore"]) == 3
