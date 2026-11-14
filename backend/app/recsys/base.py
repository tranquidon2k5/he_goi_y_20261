"""Interface chung cho mọi thuật toán gợi ý.

App chỉ phụ thuộc vào lớp `Recommender` dưới đây. Khi nhóm chốt thuật toán (SASRec,
gSASRec, XSimGCL, ...) chỉ cần viết một lớp con mới, đăng ký vào `registry.py`, rồi tạo
một bản ghi model_versions trỏ tới tên đó — không phải sửa API hay giao diện.

Quy ước:
- Người dùng được biểu diễn bằng `UserContext.history` (chuỗi tương tác theo thời gian),
  không cần embedding riêng — hợp với hướng tuần tự trong PRD mục 7, nhưng mô hình
  không tuần tự cũng dùng được (bỏ qua thứ tự).
- Recommender KHÔNG cần tự lọc phim đã xem / "Không quan tâm"; tầng service đã truyền
  `exclude` và lọc lại lần nữa sau khi nhận kết quả.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from app.recsys.catalog import Catalog

InteractionKind = Literal["rating", "mark_watched", "watch", "watchlist_add", "tag"]


@dataclass(frozen=True)
class Interaction:
    movie_id: int
    timestamp: int
    kind: InteractionKind
    rating: float | None = None


@dataclass
class UserContext:
    user_id: int
    history: list[Interaction]  # sắp theo timestamp tăng dần
    genres: set[str] = field(default_factory=set)  # thể loại chọn ở onboarding
    exclude: set[int] = field(default_factory=set)

    @property
    def n_interactions(self) -> int:
        return len({i.movie_id for i in self.history})


@dataclass
class Scored:
    movie_id: int
    score: float


class Recommender(ABC):
    """Lớp cơ sở. Một instance = một checkpoint đã nạp."""

    #: tên đăng ký trong registry, lưu ở model_versions.recommender
    name: str = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def load(self, catalog: Catalog, artifact_dir: str | None = None) -> None:
        """Nạp checkpoint / dựng chỉ mục. Gọi lại mỗi khi danh mục hoặc phiên bản đổi."""

    @abstractmethod
    def recommend(self, ctx: UserContext, k: int) -> list[Scored]:
        """Top-k phim cho người dùng (hàng "Dành cho bạn")."""

    @abstractmethod
    def similar(self, movie_id: int, k: int, exclude: set[int]) -> list[Scored]:
        """Top-k phim tương tự một phim (item-to-item)."""

    def explain(self, ctx: UserContext, movie_id: int) -> int | None:
        """Phim trong lịch sử giải thích tốt nhất cho gợi ý `movie_id` ("Vì bạn đã xem X").

        PRD mục 3: lấy phim trong lịch sử có embedding gần nhất. Mặc định không giải thích.
        """
        return None
