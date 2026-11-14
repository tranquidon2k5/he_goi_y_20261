"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { genreLine, MovieCard } from "@/components/movie-card";
import { Guard } from "@/components/providers";
import { qs, useApi } from "@/lib/api";
import { track } from "@/lib/events";
import { gvi } from "@/lib/genres";
import type { SearchResult } from "@/lib/types";

type Sort = "rel" | "avg" | "year" | "pop";
type F = { q: string; genre: string[]; y1: string; y2: string; min: number; sort: Sort; page: number };

const PAGE_SIZE = 40;

function fromParams(sp: URLSearchParams): F {
  return {
    q: sp.get("q") ?? "",
    genre: sp.getAll("genre"),
    y1: sp.get("year_min") ?? "",
    y2: sp.get("year_max") ?? "",
    min: Number(sp.get("min_avg") ?? 0),
    sort: (sp.get("sort") as Sort) ?? "rel",
    page: Number(sp.get("page") ?? 1),
  };
}

function toQuery(f: F) {
  return qs({ q: f.q.trim(), genre: f.genre, year_min: f.y1, year_max: f.y2, min_avg: f.min || null, sort: f.sort,
    page: f.page > 1 ? f.page : null });
}

function useDebounced<T>(v: T, ms: number) {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

function Search() {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const [f, setF] = useState<F>(() => fromParams(new URLSearchParams(sp.toString())));
  const genres = useApi<string[]>("/movies/genres");

  // ô tìm trên header đổi ?q= → đồng bộ vào bộ lọc (bỏ qua thay đổi URL do chính trang này đẩy lên)
  const urlQ = sp.get("q") ?? "";
  const pushedQ = useRef(urlQ);
  useEffect(() => {
    if (urlQ !== pushedQ.current) {
      pushedQ.current = urlQ;
      setF((x) => ({ ...x, q: urlQ, page: 1 }));
    }
  }, [urlQ]);

  const df = useDebounced(f, 300);
  const query = useMemo(() => toQuery(df), [df]);
  const res = useApi<SearchResult>(`/movies${query}${query ? "&" : "?"}page_size=${PAGE_SIZE}`);

  useEffect(() => {
    pushedQ.current = df.q.trim();
    router.replace(`${path}${query}`, { scroll: false });
    if (df.q || df.genre.length || df.y1 || df.y2 || df.min) {
      track("search", { payload: { q: df.q, genres: df.genre, year_min: df.y1, year_max: df.y2, min_avg: df.min, sort: df.sort } });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  const set = (p: Partial<F>) => setF((x) => ({ ...x, page: 1, ...p }));
  const active = [
    ...(f.q ? [`“${f.q}”`] : []), ...f.genre.map(gvi),
    ...(f.y1 || f.y2 ? [`${f.y1 || "…"}–${f.y2 || "…"}`] : []), ...(f.min ? [`★ ≥ ${f.min}`] : []),
  ];
  const total = res.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <main>
      <div className="search-layout">
        <aside className="filters">
          <div>
            <h3 style={{ marginBottom: 8 }}>Tìm theo tên</h3>
            <input value={f.q} onChange={(e) => set({ q: e.target.value })} placeholder="vd. Inception" style={{ width: "100%" }} aria-label="Tên phim" />
          </div>
          <div>
            <h3 style={{ marginBottom: 8 }}>Thể loại</h3>
            <div className="chips">
              {(genres.data ?? []).map((g) => {
                const on = f.genre.includes(g);
                return (
                  <button key={g} className={`chip ${on ? "on" : ""}`} aria-pressed={on}
                    onClick={() => set({ genre: on ? f.genre.filter((x) => x !== g) : [...f.genre, g] })}>{gvi(g)}</button>
                );
              })}
            </div>
          </div>
          <div>
            <h3 style={{ marginBottom: 8 }}>Năm phát hành</h3>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input type="number" value={f.y1} placeholder="từ" onChange={(e) => set({ y1: e.target.value })} style={{ width: 90 }} aria-label="Từ năm" />
              <span className="dim">–</span>
              <input type="number" value={f.y2} placeholder="đến" onChange={(e) => set({ y2: e.target.value })} style={{ width: 90 }} aria-label="Đến năm" />
            </div>
          </div>
          <div>
            <h3 style={{ marginBottom: 8 }}>Điểm trung bình tối thiểu: <span className="best">{f.min}</span></h3>
            <input type="range" min={0} max={5} step={0.5} value={f.min} onChange={(e) => set({ min: Number(e.target.value) })}
              style={{ width: "100%", padding: 0, accentColor: "#f5b544" }} aria-label="Điểm tối thiểu" />
          </div>
          <div>
            <h3 style={{ marginBottom: 8 }}>Sắp xếp</h3>
            <select value={f.sort} onChange={(e) => set({ sort: e.target.value as Sort })} style={{ width: "100%" }} aria-label="Sắp xếp">
              <option value="rel">Phù hợp với bạn</option>
              <option value="avg">Điểm trung bình</option>
              <option value="year">Mới nhất</option>
              <option value="pop">Phổ biến</option>
            </select>
          </div>
          <button className="btn ghost" onClick={() => setF({ q: "", genre: [], y1: "", y2: "", min: 0, sort: "rel", page: 1 })}>Xoá bộ lọc</button>
        </aside>
        <div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 18 }}>
            <h1>Khám phá</h1><span className="muted">{res.data ? `${total.toLocaleString("vi")} phim` : ""}</span>
          </div>
          {active.length > 0 && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16 }}>
              {active.map((a) => <span key={a} className="tag" style={{ padding: "4px 10px" }}>{a}</span>)}
            </div>
          )}
          {res.data && res.data.items.length === 0 ? (
            <div className="empty">
              <div style={{ fontSize: 28 }}>∅</div>
              <h3 style={{ marginTop: 6 }}>Không có phim nào khớp</h3>
              <p>Thử bỏ bớt thể loại hoặc hạ điểm tối thiểu.</p>
            </div>
          ) : (
            <div className="result-grid" style={{ opacity: res.loading ? 0.5 : 1 }}>
              {res.data?.items.map((it, i) => (
                <MovieCard key={it.movie.id} movie={it.movie}
                  source={{ row: "search", position: (f.page - 1) * PAGE_SIZE + i }}
                  reason={it.my_rating ? `Bạn đã chấm ${it.my_rating}★` : genreLine(it.movie, 3)} />
              ))}
            </div>
          )}
          {pages > 1 && (
            <div className="pager">
              <button className="btn" disabled={f.page <= 1} onClick={() => setF({ ...f, page: f.page - 1 })}>← Trước</button>
              <span className="muted">Trang {f.page} / {pages}</span>
              <button className="btn" disabled={f.page >= pages} onClick={() => setF({ ...f, page: f.page + 1 })}>Sau →</button>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

export function SearchView() {
  return <Guard><Search /></Guard>;
}
