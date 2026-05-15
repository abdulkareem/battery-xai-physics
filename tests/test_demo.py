from battery_xai.data.demo import generate_demo_raw_data
from battery_xai.data.schema import CANONICAL_COLUMNS


def test_demo_data_matches_canonical_schema():
    frame = generate_demo_raw_data(cells_per_dataset=1, cycles=3, points_per_cycle=2)
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert set(frame["dataset"]) == {"nasa", "calce", "oxford", "iontech"}
    assert frame["soh"].between(0, 1.1).all()
