"""Declared metadata for every feature: definition, rationale, lookback and leakage notes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str
    rationale: str
    formula: str
    lookback: int
    leakage_note: str = "Backward-looking rolling window ending at t inclusive."
    scale_free: bool = True


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def add(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate feature name: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def extend(self, specs: list[FeatureSpec]) -> None:
        for spec in specs:
            self.add(spec)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, name: str) -> FeatureSpec:
        return self._specs[name]

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def by_group(self, group: str) -> list[FeatureSpec]:
        return [s for s in self._specs.values() if s.group == group]

    def max_lookback(self, names: list[str] | None = None) -> int:
        selected = names or self.names
        return max((self._specs[n].lookback for n in selected if n in self._specs), default=0)

    def to_frame(self, names: list[str] | None = None) -> pd.DataFrame:
        selected = names or self.names
        rows = [
            {
                "feature": s.name,
                "group": s.group,
                "lookback": s.lookback,
                "description": s.description,
                "formula": s.formula,
                "rationale": s.rationale,
                "leakage_note": s.leakage_note,
            }
            for s in (self._specs[n] for n in selected if n in self._specs)
        ]
        return pd.DataFrame(rows)

    def to_markdown(self, names: list[str] | None = None) -> str:
        frame = self.to_frame(names)
        lines = ["| Feature | Group | Lookback | Definition | Rationale |", "|---|---|---|---|---|"]
        for row in frame.itertuples(index=False):
            lines.append(
                f"| `{row.feature}` | {row.group} | {row.lookback} | "
                f"{row.description}. `{row.formula}` | {row.rationale} |"
            )
        return "\n".join(lines)


@dataclass
class BuildResult:
    frame: pd.DataFrame
    specs: list[FeatureSpec] = field(default_factory=list)
