import pytest
from iteratex.preprocessing import validate_json


def test_validate_json_pass():
    record = {"feature1": 1.2, "feature2": 3.4, "label": 1}
    assert validate_json(record) == record


def test_validate_json_fail():
    bad = {"feature1": 1.0}
    with pytest.raises(ValueError):
        validate_json(bad)
