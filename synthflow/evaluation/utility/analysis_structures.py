from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Analysis:
    metric: str


@dataclass
class ColumnAnalysis(Analysis):
    metric: str
    target: str


@dataclass
class ColumnAnalysis1Way(ColumnAnalysis):
    binning: bool = False


@dataclass
class ColumnAnalysis2Way(ColumnAnalysis):
    by: str
    binning: bool = True


@dataclass
class JointAnalysis(Analysis):
    binning: bool
    targets: None | Iterable[str] = None
