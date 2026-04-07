"""Tests for timestamp module."""

import pytest

from vlfrx.core.timestamp import Timestamp


class TestTimestamp:
    def test_create_timestamp(self):
        t = Timestamp(1234567890, 0.5)
        assert t.secs == 1234567890
        assert t.frac == 0.5

    def test_normalize_positive(self):
        t = Timestamp(100, 1.5)
        assert t.secs == 101
        assert t.frac == 0.5

    def test_normalize_negative_secs(self):
        t = Timestamp(-5, 0.5)
        assert t.secs == 0
        assert t.frac == 0.0

    def test_from_seconds(self):
        t = Timestamp.from_seconds(1234567890.5)
        assert t.secs == 1234567890
        assert abs(t.frac - 0.5) < 1e-10

    def test_now(self):
        t = Timestamp.now()
        assert t.secs > 1700000000  # Should be after 2023

    def test_parse_iso8601(self):
        t = Timestamp.parse("2024-01-15T12:30:45.123456")
        assert t.secs == 1705321845  # 2024-01-15T12:30:45 UTC
        assert abs(t.frac - 0.123456) < 1e-6

    def test_parse_unix_timestamp(self):
        t = Timestamp.parse("1705321845.123456")
        assert t.secs == 1705321845
        assert abs(t.frac - 0.123456) < 1e-6

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            Timestamp.parse("not a timestamp")

    def test_to_seconds(self):
        t = Timestamp(100, 0.25)
        assert t.to_seconds() == 100.25

    def test_format(self):
        t = Timestamp(100, 0.1234567)
        assert t.format() == "100.1234567"

    def test_format_iso(self):
        t = Timestamp(1705321845, 0.123456)  # 2024-01-15T12:30:45 UTC
        result = t.format_iso()
        assert result.startswith("2024-01-15T12:30:45")

    def test_equality(self):
        t1 = Timestamp(100, 0.5)
        t2 = Timestamp(100, 0.5)
        assert t1 == t2

    def test_comparison(self):
        t1 = Timestamp(100, 0.5)
        t2 = Timestamp(101, 0.0)
        assert t1 < t2
        assert t2 > t1

    def test_addition(self):
        t1 = Timestamp(100, 0.3)
        t2 = Timestamp(50, 0.5)
        result = t1 + t2
        assert result.secs == 150
        assert abs(result.frac - 0.8) < 1e-10

    def test_subtraction(self):
        t1 = Timestamp(100, 0.8)
        t2 = Timestamp(50, 0.3)
        result = t1 - t2
        assert result.secs == 50
        assert abs(result.frac - 0.5) < 1e-10

    def test_is_zero(self):
        assert Timestamp.ZERO.is_zero()
        t = Timestamp(1, 0)
        assert not t.is_zero()

    def test_is_none(self):
        assert Timestamp.NONE.is_none()
        t = Timestamp(0, 0)
        assert not t.is_none()

    def test_repr(self):
        t = Timestamp(100, 0.5)
        assert repr(t) == "Timestamp(100, 0.5)"

    def test_str(self):
        t = Timestamp(100, 0.5)
        assert str(t) == "100.5000000"

    def test_hash(self):
        t1 = Timestamp(100, 0.5)
        t2 = Timestamp(100, 0.5)
        assert hash(t1) == hash(t2)


class TestTimestampFromFilename:
    def test_parse_standard_format(self):
        t = Timestamp.from_filename("signal_20240115_123045.vt")
        assert t.secs > 0

    def test_parse_with_fraction(self):
        t = Timestamp.from_filename("signal_20240115_123045.123.vt")
        assert t.secs > 0
        assert t.frac > 0

    def test_no_timestamp(self):
        t = Timestamp.from_filename("random.vt")
        assert t.is_zero()
