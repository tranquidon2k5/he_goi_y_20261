from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CINEREC_", extra="ignore")

    # SQLite cho dev; production dùng postgresql+psycopg://user:pass@host/db
    database_url: str = "sqlite:///./cinerec.db"
    jwt_secret: str = "dev-only-secret-change-me-in-production-0000"
    jwt_ttl_hours: int = 24 * 14
    cookie_secure: bool = False

    # PRD mục 4: userId của app bắt đầu từ 1.000.001 để không trùng MovieLens
    app_user_id_start: int = 1_000_001
    # Phim do quản trị thêm nhận movieId từ 10.000.001 để không trùng movieId của MovieLens
    app_movie_id_start: int = 10_000_001
    # PRD mục 3: dưới ngưỡng này dùng "Thịnh hành" theo thể loại (switching hybrid)
    cold_start_threshold: int = 5
    positive_rating: float = 4.0
    # Cửa sổ tính "Thịnh hành" (giây), tính lùi từ timestamp mới nhất trong ratings
    trending_window_seconds: int = 365 * 24 * 3600

    tmdb_api_key: str | None = None
    admin_emails: str = ""  # danh sách email admin, phân tách bằng dấu phẩy


@lru_cache
def get_settings() -> Settings:
    return Settings()
