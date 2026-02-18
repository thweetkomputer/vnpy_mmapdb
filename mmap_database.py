"""Memory-mapped database backend."""

from __future__ import annotations

import math
import mmap
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import (
    DB_TZ,
    BarOverview,
    BaseDatabase,
    TickOverview,
    convert_tz,
)
from vnpy.trader.object import BarData, TickData
from vnpy.trader.setting import SETTINGS


BASE_DATETIME: datetime = datetime(2021, 1, 1, tzinfo=DB_TZ).replace(tzinfo=None)
BASE_DATE = BASE_DATETIME.date()
INTERVAL_SECONDS: dict[Interval, int] = {
    Interval.MINUTE: 60,
    Interval.HOUR: 3600,
    Interval.DAILY: 86400,
    Interval.WEEKLY: 604800,
}
INTERVAL_ALIASES: dict[Interval, str] = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
    Interval.WEEKLY: "1w",
}
ALIAS_TO_INTERVAL: dict[str, Interval] = {}
for interval, alias in INTERVAL_ALIASES.items():
    ALIAS_TO_INTERVAL[alias] = interval
    ALIAS_TO_INTERVAL[interval.value] = interval

TRADING_MINUTE_SEGMENTS: list[tuple[int, int]] = [
    (9 * 60 + 31, 11 * 60 + 30),
    (13 * 60 + 1, 15 * 60 + 0),
]
TRADING_MINUTES_PER_DAY: int = sum(
    end - start + 1 for start, end in TRADING_MINUTE_SEGMENTS
)

BAR_STRUCT: struct.Struct = struct.Struct("<q i i i i q q q")
BAR_RECORD_SIZE: int = BAR_STRUCT.size
BAR_DTYPE = np.dtype(
    [
        ("offset_sec", "<i8"),
        ("open_price", "<i4"),
        ("high_price", "<i4"),
        ("low_price", "<i4"),
        ("close_price", "<i4"),
        ("volume", "<i8"),
        ("turnover", "<i8"),
        ("open_interest", "<i8"),
    ]
)
EMPTY_BAR_RECORD: bytes = BAR_STRUCT.pack(-1, 0, 0, 0, 0, 0, 0, 0)
PRICE_SCALE: int = max(int(SETTINGS.get("mmapdb.price_scale", 1)), 1)


@dataclass(slots=True)
class BarSlot:
    """Decoded bar record."""

    offset_sec: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    turnover: float
    open_interest: float


class Database(BaseDatabase):
    """Memory-mapped file database implementation."""

    def __init__(self) -> None:
        storage_setting: str = SETTINGS.get("database.database", "database.db")
        storage_path = Path(storage_setting).expanduser()
        if storage_path.is_file():
            storage_path = storage_path.parent / storage_path.stem
        elif storage_path.suffix:
            storage_path = storage_path.with_suffix("")

        self.base_path: Path = storage_path
        if self.base_path.name != "mmapdb":
            self.base_path = self.base_path / "mmapdb"

        if self.base_path.exists() and not self.base_path.is_dir():
            raise RuntimeError(
                f"mmapdb 路径 {self.base_path} 是文件，请改为目录或删除原 SQLite 文件"
            )
        self.bar_path: Path = self.base_path / "bars"
        self.tick_path: Path = self.base_path / "ticks"

        self.bar_path.mkdir(parents=True, exist_ok=True)
        self.tick_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    def save_bar_data(self, bars: list[BarData], stream: bool = False) -> bool:
        if not bars:
            return True

        batches: dict[Path, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)

        for bar in bars:
            if not bar.interval:
                continue

            seconds: int | None = INTERVAL_SECONDS.get(bar.interval)
            if not seconds:
                continue

            dt: datetime = convert_tz(bar.datetime)
            if dt < BASE_DATETIME:
                continue

            offset_sec: int = int((dt - BASE_DATETIME).total_seconds())
            if bar.interval == Interval.MINUTE:
                slot_index = self._minute_slot_index(dt)
                if slot_index is None:
                    continue
            else:
                slot_index = max(offset_sec // seconds, 0)

            path: Path = self._bar_file_path(bar.symbol, bar.exchange, bar.interval)

            record = (
                offset_sec,
                self._encode_price(bar.open_price),
                self._encode_price(bar.high_price),
                self._encode_price(bar.low_price),
                self._encode_price(bar.close_price),
                self._encode_int(bar.volume),
                self._encode_int(bar.turnover),
                self._encode_int(bar.open_interest),
            )

            batches[path].append((slot_index, record))

        for path, entries in batches.items():
            self._write_records(path, entries)

        return True

    # ----------------------------------------------------------------------
    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
        # Tick data is not stored in mmap database for now.
        return True

    # ----------------------------------------------------------------------
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        seconds: int | None = INTERVAL_SECONDS.get(interval)
        if not seconds:
            return []

        path: Path = self._bar_file_path(symbol, exchange, interval)
        if not path.exists():
            return []

        start_dt: datetime = convert_tz(start)
        end_dt: datetime = convert_tz(end)
        if end_dt <= start_dt:
            return []

        if interval == Interval.MINUTE:
            start_slot, end_slot = self._minute_slot_range(start_dt, end_dt)
        else:
            start_slot = max(
                int((start_dt - BASE_DATETIME).total_seconds()) // seconds,
                0,
            )
            end_slot = max(
                int(math.ceil((end_dt - BASE_DATETIME).total_seconds() / seconds)),
                0,
            )

        records: list[BarSlot] = list(
            self._read_bar_range(path, start_slot, end_slot)
        )

        bars: list[BarData] = []
        for record in records:
            bar_dt: datetime = BASE_DATETIME + timedelta(seconds=record.offset_sec)
            if bar_dt < start_dt or bar_dt > end_dt:
                continue

            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=bar_dt,
                interval=interval,
                gateway_name="MMAP",
                open_price=record.open_price,
                high_price=record.high_price,
                low_price=record.low_price,
                close_price=record.close_price,
                volume=record.volume,
                turnover=record.turnover,
                open_interest=record.open_interest,
            )
            bars.append(bar)

        return bars

    # ----------------------------------------------------------------------
    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
    ) -> list[TickData]:
        return []

    # ----------------------------------------------------------------------
    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> int:
        path: Path = self._bar_file_path(symbol, exchange, interval)
        if not path.exists():
            return 0

        count: int = self._count_bar_records(path)
        path.unlink(missing_ok=True)
        return count

    # ----------------------------------------------------------------------
    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
    ) -> int:
        return 0

    # ----------------------------------------------------------------------
    def get_bar_overview(self) -> list[BarOverview]:
        overviews: list[BarOverview] = []
        for exchange_dir in self.bar_path.iterdir():
            if not exchange_dir.is_dir():
                continue

            exchange: Exchange | None = self._parse_exchange(exchange_dir.name)
            if not exchange:
                continue

            for path in exchange_dir.glob("*.mmap"):
                parsed = self._parse_bar_filename(path.name)
                if not parsed:
                    continue

                symbol, interval = parsed
                if not interval:
                    continue

                slot_info = self._collect_bar_overview(path)
                overviews.append(
                    BarOverview(
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        count=slot_info["count"],
                        start=slot_info["start"],
                        end=slot_info["end"],
                    )
                )

        return overviews

    # ----------------------------------------------------------------------
    def get_tick_overview(self) -> list[TickOverview]:
        return []

    # ----------------------------------------------------------------------
    def _bar_file_path(
        self, symbol: str, exchange: Exchange, interval: Interval
    ) -> Path:
        folder = self.bar_path / exchange.value
        folder.mkdir(parents=True, exist_ok=True)
        alias = INTERVAL_ALIASES.get(interval, interval.value)
        filename = f"{self._sanitize_symbol(symbol)}_{alias}.mmap"
        return folder / filename

    # ----------------------------------------------------------------------
    def _read_bar_range(
        self, path: Path, start_slot: int, end_slot: int
    ) -> Iterable[BarSlot]:
        with open(path, "rb") as file_:
            size = os.path.getsize(path)
            if size == 0:
                return

            record_total = size // BAR_RECORD_SIZE
            if start_slot >= record_total:
                return

            end_slot = min(end_slot, record_total)
            with mmap.mmap(file_.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                start_off = start_slot * BAR_RECORD_SIZE
                end_off = end_slot * BAR_RECORD_SIZE
                view = memoryview(mm)[start_off:end_off]
                if not view:
                    return

                raw = np.frombuffer(view, dtype=BAR_DTYPE).copy()
                del view
                if not len(raw):
                    return

                for row in raw:
                    if int(row["offset_sec"]) < 0:
                        continue

                    yield BarSlot(
                        offset_sec=int(row["offset_sec"]),
                        open_price=self._decode_price(int(row["open_price"])),
                        high_price=self._decode_price(int(row["high_price"])),
                        low_price=self._decode_price(int(row["low_price"])),
                        close_price=self._decode_price(int(row["close_price"])),
                        volume=float(row["volume"]),
                        turnover=float(row["turnover"]),
                        open_interest=float(row["open_interest"]),
                    )

    # ----------------------------------------------------------------------
    def _write_records(
        self, path: Path, entries: list[tuple[int, tuple[int, ...]]]
    ) -> None:
        if not entries:
            return

        entries.sort(key=lambda item: item[0])
        max_slot: int = entries[-1][0]
        required_records = max_slot + 1
        self._resize_file(path, required_records, BAR_RECORD_SIZE, EMPTY_BAR_RECORD)

        slots = np.fromiter((slot for slot, _ in entries), dtype=np.int64)
        records = np.array([record for _, record in entries], dtype=BAR_DTYPE)

        segments: list[tuple[int, bytes]] = []
        start_idx = 0
        for idx in range(1, len(slots) + 1):
            if idx == len(slots) or slots[idx] != slots[idx - 1] + 1:
                segment_bytes = records[start_idx:idx].tobytes()
                segments.append((int(slots[start_idx]), segment_bytes))
                start_idx = idx

        with open(path, "r+b") as file_:
            with mmap.mmap(file_.fileno(), 0) as mm:
                for slot_index, payload in segments:
                    offset = slot_index * BAR_RECORD_SIZE
                    mm[offset : offset + len(payload)] = payload
                mm.flush()

    # ----------------------------------------------------------------------
    def _resize_file(
        self, path: Path, required_records: int, record_size: int, empty_record: bytes
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+b") as file_:
            file_.seek(0, os.SEEK_END)
            current_size = file_.tell()
            current_records = current_size // record_size

            if required_records <= current_records:
                return

            file_.truncate(required_records * record_size)
            file_.flush()

            file_.seek(current_records * record_size)
            append_count = required_records - current_records
            if append_count > 0:
                file_.write(empty_record * append_count)
                file_.flush()

    # ----------------------------------------------------------------------
    def _parse_bar_filename(self, name: str) -> tuple[str, Interval | None] | None:
        stem = Path(name).stem
        if "_" not in stem:
            return None

        symbol, alias = stem.rsplit("_", maxsplit=1)
        interval = ALIAS_TO_INTERVAL.get(alias)
        return symbol, interval

    # ----------------------------------------------------------------------
    def _collect_bar_overview(self, path: Path) -> dict[str, datetime | int | None]:
        start_dt: datetime | None = None
        end_dt: datetime | None = None
        count: int = 0

        with open(path, "rb") as file_:
            size = os.path.getsize(path)
            if not size:
                return {"start": None, "end": None, "count": 0}

            with mmap.mmap(file_.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = memoryview(mm)
                raw = np.frombuffer(data, dtype=BAR_DTYPE).copy()
                del data
                if not len(raw):
                    return {"start": None, "end": None, "count": 0}

                valid_mask = raw["offset_sec"] >= 0
                if not np.any(valid_mask):
                    return {"start": None, "end": None, "count": 0}

                offsets = raw["offset_sec"][valid_mask].astype(np.int64)
                start_dt = BASE_DATETIME + timedelta(seconds=int(offsets.min()))
                end_dt = BASE_DATETIME + timedelta(seconds=int(offsets.max()))
                count = int(np.count_nonzero(valid_mask))

        return {"start": start_dt, "end": end_dt, "count": count}

    # ----------------------------------------------------------------------
    def _count_bar_records(self, path: Path) -> int:
        if not path.exists():
            return 0

        total = 0
        with open(path, "rb") as file_:
            size = os.path.getsize(path)
            if not size:
                return 0

            with mmap.mmap(file_.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = memoryview(mm)
                raw = np.frombuffer(data, dtype=BAR_DTYPE).copy()
                del data
                if not len(raw):
                    return 0

                total = int(np.count_nonzero(raw["offset_sec"] >= 0))

        return total

    # ----------------------------------------------------------------------
    def _sanitize_symbol(self, symbol: str) -> str:
        return symbol.replace(os.sep, "_").replace(".", "_")

    # ----------------------------------------------------------------------
    def _parse_exchange(self, value: str) -> Exchange | None:
        try:
            return Exchange(value)
        except ValueError:
            return None

    # ----------------------------------------------------------------------
    @staticmethod
    def _minute_slot_index(dt: datetime) -> int | None:
        day_offset = (dt.date() - BASE_DATE).days
        if day_offset < 0:
            return None

        minute_offset = Database._minute_offset_in_day(dt)
        if minute_offset is None:
            return None

        return day_offset * TRADING_MINUTES_PER_DAY + minute_offset

    # ----------------------------------------------------------------------
    @staticmethod
    def _minute_slot_range(start_dt: datetime, end_dt: datetime) -> tuple[int, int]:
        start_day = (start_dt.date() - BASE_DATE).days
        end_day = (end_dt.date() - BASE_DATE).days

        start_slot = max(start_day * TRADING_MINUTES_PER_DAY, 0)
        end_slot = max((end_day + 1) * TRADING_MINUTES_PER_DAY, 0)

        return start_slot, end_slot

    # ----------------------------------------------------------------------
    @staticmethod
    def _minute_offset_in_day(dt: datetime) -> int | None:
        minute_value = dt.hour * 60 + dt.minute
        offset = 0
        for start, end in TRADING_MINUTE_SEGMENTS:
            if start <= minute_value <= end:
                return offset + (minute_value - start)
            offset += end - start + 1

        return None

    # ----------------------------------------------------------------------
    @staticmethod
    def _encode_price(value: float) -> int:
        return int(round(value * PRICE_SCALE))

    # ----------------------------------------------------------------------
    @staticmethod
    def _decode_price(value: int) -> float:
        return value / PRICE_SCALE

    # ----------------------------------------------------------------------
    @staticmethod
    def _encode_int(value: float) -> int:
        return int(round(value))
