"use client";

import Link from "next/link";
import { useState } from "react";
import { Guard, useToast } from "@/components/providers";
import { Poster } from "@/components/poster";
import { api, useApi } from "@/lib/api";
import { gvi } from "@/lib/genres";
import type { AdminMovies, Movie } from "@/lib/types";

type Form = {
  title: string; year: string; genres: string[]; overview: string; tmdb_id: string; imdb_id: string;
  trailer_key: string; runtime: string; directors: string; cast: string; fetch_tmdb: boolean;
};
const EMPTY: Form = { title: "", year: "", genres: [], overview: "", tmdb_id: "", imdb_id: "", trailer_key: "",
  runtime: "", directors: "", cast: "", fetch_tmdb: false };

const fromMovie = (m: Movie): Form => ({
  title: m.title, year: m.year?.toString() ?? "", genres: m.genres, overview: m.overview ?? "",
  tmdb_id: m.tmdb_id?.toString() ?? "", imdb_id: m.imdb_id ?? "", trailer_key: m.trailer_key ?? "",
  runtime: m.runtime?.toString() ?? "", directors: m.directors ?? "", cast: m.cast ?? "", fetch_tmdb: false,
});

function payload(f: Form) {
  const int = (s: string) => (s.trim() ? Number(s) : null);
  const str = (s: string) => s.trim() || null;
  return {
    title: f.title.trim(), year: int(f.year), genres: f.genres, overview: str(f.overview), tmdb_id: int(f.tmdb_id),
    imdb_id: str(f.imdb_id), trailer_key: str(f.trailer_key), runtime: int(f.runtime), directors: str(f.directors),
    cast: str(f.cast), fetch_tmdb: f.fetch_tmdb,
  };
}

function MovieAdmin() {
  const toast = useToast();
  const [page, setPage] = useState(1);
  const { data, reload } = useApi<AdminMovies>(`/admin/movies?page=${page}`);
  const { data: allGenres } = useApi<string[]>("/movies/genres");
  const [form, setForm] = useState<Form>(EMPTY);
  const [editing, setEditing] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  const toggleGenre = (g: string) => set("genres", form.genres.includes(g) ? form.genres.filter((x) => x !== g) : [...form.genres, g]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.genres.length) return setErr("Chọn ít nhất một thể loại — mô hình dùng thể loại để xếp hạng phim mới.");
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ movie: Movie; warning: string | null }>(editing ? `/admin/movies/${editing}` : "/admin/movies", {
        method: editing ? "PATCH" : "POST", json: payload(form),
      });
      toast(editing ? `Đã cập nhật “${r.movie.title}”` : `Đã thêm “${r.movie.title}” · movieId ${r.movie.id}`);
      if (r.warning) setErr(`Đã lưu, nhưng: ${r.warning}`);
      setForm(EMPTY);
      setEditing(null);
      setPage(1);
      reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Có lỗi xảy ra");
    } finally {
      setBusy(false);
    }
  }

  async function remove(m: Movie) {
    if (!confirm(`Xoá “${m.title}”? Rating và danh sách của người dùng với phim này cũng bị xoá.`)) return;
    await api(`/admin/movies/${m.id}`, { method: "DELETE" });
    toast(`Đã xoá “${m.title}”`);
    if (editing === m.id) { setEditing(null); setForm(EMPTY); }
    reload();
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <main>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <h1>Quản trị phim</h1><span className="tag">Chỉ admin</span>
      </div>
      <p className="muted" style={{ marginTop: 6, maxWidth: 760 }}>
        Phim thêm ở đây vào hàng “Mới thêm vào kho” ngay. Mô hình gợi ý xếp hạng nó nhờ đặc trưng nội dung
        (thể loại, năm) cho tới lần fine-tune kế tiếp ở <Link href="/lab">Lab</Link>, khi embedding riêng của phim được học từ tương tác.
      </p>

      <div className="admin-grid">
        <form className="card admin-form" onSubmit={submit}>
          <h2>{editing ? `Sửa phim #${editing}` : "Thêm phim mới"}</h2>
          <label>Tên phim *
            <input required maxLength={500} value={form.title} onChange={(e) => set("title", e.target.value)} />
          </label>
          <div className="two">
            <label>Năm
              <input type="number" min={1880} max={2100} value={form.year} onChange={(e) => set("year", e.target.value)} />
            </label>
            <label>Thời lượng (phút)
              <input type="number" min={1} max={1000} value={form.runtime} onChange={(e) => set("runtime", e.target.value)} />
            </label>
          </div>
          <fieldset>
            <legend>Thể loại *</legend>
            <div className="chips">
              {(allGenres ?? []).map((g) => (
                <button type="button" key={g} className={`chip ${form.genres.includes(g) ? "on" : ""}`}
                  aria-pressed={form.genres.includes(g)} onClick={() => toggleGenre(g)}>{gvi(g)}</button>
              ))}
            </div>
          </fieldset>
          <label>Mô tả
            <textarea rows={4} maxLength={5000} value={form.overview} onChange={(e) => set("overview", e.target.value)} />
          </label>
          <div className="two">
            <label>Đạo diễn
              <input maxLength={500} value={form.directors} onChange={(e) => set("directors", e.target.value)} />
            </label>
            <label>Diễn viên
              <input maxLength={1000} value={form.cast} onChange={(e) => set("cast", e.target.value)} />
            </label>
          </div>
          <div className="two">
            <label>TMDB id
              <input type="number" min={1} value={form.tmdb_id} onChange={(e) => set("tmdb_id", e.target.value)} />
            </label>
            <label>IMDb id
              <input maxLength={20} placeholder="tt1234567" value={form.imdb_id} onChange={(e) => set("imdb_id", e.target.value)} />
            </label>
          </div>
          <label>Mã trailer YouTube
            <input maxLength={50} placeholder="vd. dQw4w9WgXcQ" value={form.trailer_key} onChange={(e) => set("trailer_key", e.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={form.fetch_tmdb} disabled={!form.tmdb_id}
              onChange={(e) => set("fetch_tmdb", e.target.checked)} />
            Lấy mô tả, poster, trailer, đạo diễn, diễn viên từ TMDB theo TMDB id
            {data && !data.tmdb_enabled && <span className="dim"> (chưa cấu hình CINEREC_TMDB_API_KEY)</span>}
          </label>
          {err && <div className="error" role="alert">{err}</div>}
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn primary" disabled={busy}>{busy ? "Đang lưu…" : editing ? "Lưu thay đổi" : "Thêm phim"}</button>
            {editing && <button type="button" className="btn" onClick={() => { setEditing(null); setForm(EMPTY); setErr(null); }}>Huỷ</button>}
          </div>
        </form>

        <div className="card">
          <h2>Phim đã thêm {data ? <span className="dim">· {data.total}</span> : null}</h2>
          {!data ? <div className="skeleton" style={{ height: 200, marginTop: 12 }} /> : data.items.length ? (
            <ul className="admin-list">
              {data.items.map((m) => (
                <li key={m.id}>
                  <Link href={`/movies/${m.id}`} className="mini-poster"><Poster movie={m} sizes="56px" /></Link>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <Link href={`/movies/${m.id}`} className="t">{m.title}</Link>
                    <div className="dim" style={{ fontSize: 12.5 }}>
                      #{m.id} · {m.year ?? "—"} · {m.genres.map(gvi).join(", ")} · {m.rating_count} lượt chấm
                    </div>
                  </div>
                  <button className="iconbtn" onClick={() => { setEditing(m.id); setForm(fromMovie(m)); setErr(null); window.scrollTo({ top: 0, behavior: "smooth" }); }}>Sửa</button>
                  <button className="iconbtn" onClick={() => remove(m)}>Xoá</button>
                </li>
              ))}
            </ul>
          ) : <div className="empty" style={{ marginTop: 12, padding: 28 }}>Chưa có phim nào do quản trị thêm.</div>}
          {pages > 1 && (
            <div className="pager">
              <button className="btn" disabled={page <= 1} onClick={() => setPage(page - 1)}>‹</button>
              <span className="muted">{page} / {pages}</span>
              <button className="btn" disabled={page >= pages} onClick={() => setPage(page + 1)}>›</button>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

export default function AdminPage() {
  return <Guard admin><MovieAdmin /></Guard>;
}
