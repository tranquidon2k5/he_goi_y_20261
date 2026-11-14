"""Đăng ký các Recommender. Thêm mô hình mới: import lớp của nó và thêm vào RECOMMENDERS."""

from app.recsys.base import Recommender
from app.recsys.heuristic import HeuristicRecommender

RECOMMENDERS: dict[str, type[Recommender]] = {
    HeuristicRecommender.name: HeuristicRecommender,
}

DEFAULT_RECOMMENDER = HeuristicRecommender.name


def create(name: str, config: dict | None = None) -> Recommender:
    try:
        cls = RECOMMENDERS[name]
    except KeyError as e:
        raise ValueError(f"Recommender chưa đăng ký: {name!r}. Có: {sorted(RECOMMENDERS)}") from e
    return cls(config)
