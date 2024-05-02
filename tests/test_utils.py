import pandas as pd
import pytest

from synthflow.utils import (
    compute_complete_counts,
    get_single_row_by_query,
    interval_number,
    n_way_gen,
)


def test_n_way_gen():
    """Test generating of all n-way tuples."""
    assert list(n_way_gen("abcd", [1, 2, 3])) == [
        ("a",),
        ("b",),
        ("c",),
        ("d",),
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "c", "d"),
        ("b", "c", "d"),
    ]


def test_compute_complete_counts():
    """Test count completion of a missing index."""

    counts1 = pd.Series({"a": 1, "b": 2})
    counts2 = pd.Series({"a": 3, "c": 4})

    counts1_complete, counts2_complete = compute_complete_counts(counts1, counts2, 0)

    assert counts1_complete.to_dict() == {"a": 1, "b": 2, "c": 0}
    assert counts2_complete.to_dict() == {"a": 3, "b": 0, "c": 4}


def test_interval_number():
    """Test single number interval creation."""

    interval = interval_number(5)
    assert not interval.is_empty
    assert interval.left == interval.mid == interval.right


def test_get_single_row_by_query():
    "Test"
    df = pd.DataFrame(
        {"col1": [1, 1, 3, 3], "col2": [11, 12, 11, 12], "col3": list("abcd")}
    )

    assert get_single_row_by_query(df, "col1 == 1 & col2 == 11").to_dict() == {
        "col1": 1,
        "col2": 11,
        "col3": "a",
    }

    with pytest.raises(Exception) as e:
        assert get_single_row_by_query(df, "col1 == 1 & col2 == 13")
    assert e.type == AssertionError

    with pytest.raises(Exception) as e:
        assert get_single_row_by_query(df, "col1 == 3")
    assert e.type == AssertionError
