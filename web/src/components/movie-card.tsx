"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { gvi } from "@/lib/genres";
import { rememberSource, track, type ClickSource } from "@/lib/events";
import type { Movie, RecItem } from "@/lib/types";
import { Poster } from "./poster";

export function MovieCard({ movie, reason, rank, source, onHide }: {
  movie: Movie;
  reason?: React.ReactNode;
  rank?: number;
  source?: ClickSource;
  onHide?: (id: number) => void;
}) {
  return (
    <Link href={`/movies/${movie.id}`} className="mcard" onClick={() => source && rememberSource(movie.id, source)}>
      {rank && <div className="rank" aria-hidden>{rank}</div>}
      <Poster movie={movie} />
      {onHide && (
        <button className="x" title="Không quan tâm" aria-label={`Không quan tâm ${movie.title}`}
          onClick={(e) => { e.preventDefault(); e.stopPropagation(); onHide(movie.id); }}>✕</button>
      )}
      <div className="meta">
        {movie.is_new ? <span className="tag" style={{ color: "var(--info)" }}>Mới</span>
          : movie.rating_count > 0 ? <span style={{ color: "var(--accent)" }}>★ {movie.rating_avg.toFixed(1)}</span> : null}
        <span>{movie.year ?? ""}</span>
      </div>
      {/* poster thay thế đã in sẵn tên phim */}
      {movie.poster_url && <div className="ttl">{movie.title}</div>}
      {reason && <div className="reason">{reason}</div>}
    </Link>
  );
}

/** Một hàng phim cuộn ngang; ghi sự kiện impression một lần khi hàng hiện trên màn hình. */
export function Row({ rowKey, title, subtitle, items, ranked, requestId, modelVersion, moreHref, onHide, reasonOf }: {
  rowKey: string;
  title: string;
  subtitle?: string;
  items: RecItem[];
  ranked?: boolean;
  requestId?: string;
  modelVersion?: string;
  moreHref?: string;
  onHide?: (id: number) => void;
  reasonOf?: (it: RecItem) => React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const sent = useRef<string | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !items.length || sent.current === requestId) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && sent.current !== requestId) {
        sent.current = requestId ?? "x";
        track("impression", {
          request_id: requestId, model_version: modelVersion,
          payload: { row: rowKey, movie_ids: items.map((i) => i.movie.id), positions: items.map((_, i) => i) },
        });
        io.disconnect();
      }
    }, { threshold: 0.4 });
    io.observe(el);
    return () => io.disconnect();
  }, [items, requestId, modelVersion, rowKey]);

  return (
    <section className="row" ref={ref}>
      <div className="row-head">
        <h2>{title}</h2>
        {subtitle && <span className="sub">{subtitle}</span>}
        {moreHref && <Link className="more" href={moreHref}>Xem tất cả →</Link>}
      </div>
      <div className={`strip ${ranked ? "ranked" : ""}`}>
        {items.length ? items.map((it, i) => (
          <MovieCard key={it.movie.id} movie={it.movie} rank={ranked ? i + 1 : undefined}
            reason={reasonOf?.(it)} onHide={onHide}
            source={{ row: rowKey, position: i, request_id: requestId, model_version: modelVersion }} />
        )) : <div className="dim">Không còn phim nào trong hàng này.</div>}
      </div>
    </section>
  );
}

export const genreLine = (m: Movie, n = 2) => m.genres.slice(0, n).map(gvi).join(" · ");
