from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class SelectionPlan:
    target_ds: int
    target_didx: int
    loading_days: int
    raw_ndays: int
    ndays: int
    validinsts: torch.Tensor | None = None
    selected_alpha_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SelectionModule(Protocol):
    select_days: int | None

    def build_train_plan(self, ds: int, loader, context: dict[str, Any]) -> SelectionPlan:
        ...

    def before_fit(self, dataset, plan: SelectionPlan) -> None:
        ...

    def after_fit(self, model, plan: SelectionPlan) -> None:
        ...


class DefaultSelectionModule:
    def __init__(self, max_train_days: int = 2000, select_days: int | None = None):
        self.max_train_days = int(max_train_days)
        self.select_days = None if select_days is None else int(select_days)

    def build_train_plan(self, ds: int, loader, context: dict[str, Any]) -> SelectionPlan:
        train_delay = int(context["trainDelay"])
        ret_days = int(context["retDays"])
        ts_days = int(context["tsDays"])
        target_ds = int(context["prev_date"](ds, train_delay))
        target_didx = loader.date2didx(target_ds)
        if target_didx - ret_days < loader.data_start_didx:
            raise ValueError(f"not enough data to train for ds={ds}")
        loading_days = target_didx - loader.data_start_didx + 1
        raw_ndays = loading_days - (ret_days + train_delay)
        ndays = min(raw_ndays, self.max_train_days)
        if ndays < ts_days:
            raise ValueError(f"not enough training window for ds={ds}")
        return SelectionPlan(
            target_ds=target_ds,
            target_didx=target_didx,
            loading_days=loading_days,
            raw_ndays=raw_ndays,
            ndays=ndays,
            metadata={
                "max_train_days": self.max_train_days,
                "select_days": self.select_days,
            },
        )

    def before_fit(self, dataset, plan: SelectionPlan) -> None:
        return None

    def after_fit(self, model, plan: SelectionPlan) -> None:
        return None
