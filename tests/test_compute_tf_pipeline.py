"""Test cases for the pipeline-based implementation"""

import pytest

from termfrequency import compute_tf_pipeline


# All additional test cases, with the goal of achieving high coverage


def test_read_file_populates_data():
    """Checks that the reading of the file works"""
    collected_data = compute_tf_pipeline.read_file("inputs/input.txt")
    assert collected_data


def test_pattern():
    """Checks that the pattern is non alphanumerical"""
    collected_pattern = compute_tf_pipeline.filter_chars_and_normalize(
        "inputs/input.txt"
    )
    assert collected_pattern


def test_stop_words():
    """Checks that the stop words are removed"""
    word_list = ["White", "tigers", "live", "mostly", "in", "India"]
    collected_words = compute_tf_pipeline.remove_stop_words(word_list)
    assert collected_words == ["White", "tigers", "live", "mostly", "India"]


def test_sort():
    """Checks that the pattern is non alphanumerical"""
    collected_pattern = compute_tf_pipeline.filter_chars_and_normalize(
        "inputs/input.txt"
    )
    assert collected_pattern


# Whenever possible, please use parameterized testing for your tests


@pytest.mark.parametrize(
    "input_string,expected_count",
    [("hello world", 2), ("hello world example", 3), ("", 0), (" ", 0), (" ", 0)],
)
def test_scan_splits_string_correctly(input_string, expected_count):
    """Checks that scan function finds the correct number of words in the String"""
    assert len(compute_tf_pipeline.scan(input_string)) == expected_count
