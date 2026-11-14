from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models import Movie

TMDB_IMG = "https://image.tmdb.org/t/p"


class MovieOut(BaseModel):
    id: int
    title: str
    year: int | None
    genres: list[str]
    overview: str | None = None
    poster_url: str | None = None
    backdrop_url: str | None = None
    trailer_key: str | None = None
    runtime: int | None = None
    directors: str | None = None
    cast: str | None = None
    rating_avg: float
    rating_count: int
    is_new: bool
    imdb_id: str | None = None
    tmdb_id: int | None = None

    @classmethod
    def of(cls, m: Movie) -> "MovieOut":
        return cls(
            id=m.id, title=m.title, year=m.year, genres=m.genre_list, overview=m.overview,
            poster_url=f"{TMDB_IMG}/w342{m.poster_path}" if m.poster_path else None,
            backdrop_url=f"{TMDB_IMG}/w1280{m.backdrop_path}" if m.backdrop_path else None,
            trailer_key=m.trailer_key, runtime=m.runtime, directors=m.directors, cast=m.cast,
            rating_avg=round(m.rating_avg or 0, 3), rating_count=m.rating_count or 0,
            is_new=m.source == "app", imdb_id=m.imdb_id, tmdb_id=m.tmdb_id,
        )


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    is_admin: bool
    onboarded: bool
    genres: list[str]
    n_ratings: int


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field(min_length=1, max_length=100)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdateIn(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=100)
    genres: list[str] | None = None
    onboarded: bool | None = None


class RatingIn(BaseModel):
    movie_id: int
    rating: float

    @field_validator("rating")
    @classmethod
    def half_steps(cls, v: float) -> float:
        if not (0.5 <= v <= 5.0) or (v * 2) != int(v * 2):
            raise ValueError("rating phải trong 0.5–5, bước 0.5")
        return v


class TagIn(BaseModel):
    tag: str = Field(min_length=1, max_length=100)


class EventIn(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    type: str = Field(max_length=40)
    timestamp: int | None = None
    session_id: str | None = Field(default=None, max_length=64)
    movie_id: int | None = None
    request_id: str | None = Field(default=None, max_length=64)
    model_version: str | None = Field(default=None, max_length=50)
    payload: dict = Field(default_factory=dict)


class EventsIn(BaseModel):
    events: list[EventIn] = Field(max_length=500)
