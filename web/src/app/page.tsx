"use client";

import Link from "next/link";
import { genreLine, Row } from "@/components/movie-card";
import { Poster } from "@/components/poster";
import { Guard, useToast } from "@/components/providers";
import { api, useApi } from "@/lib/api";
import { rememberSource } from "@/lib/events";
import { gvi, rowTitle } from "@/lib/genres";
import type { HomeRecs, RecItem } from "@/lib/types";

const rowMore = (kind: string, title: string) =>
  kind === "genre" ? `/search?genre=${encodeURIComponent(title)}&sort=rel` : kind === "trending" ? "/search?sort=pop" : "/search?sort=rel";

function Home() {
  const { data, loading, error, reload, setData } = useApi<HomeRecs>("/recommendations?surface=home");
  const toast = useToast();

  async function hide(id: number) {
    // ẩn ngay trên giao diện, rồi tải lại để nhận danh sách mới từ server
    setData((d) => d && { ...d, rows: d.rows.map((r) => ({ ...r, items: r.items.filter((i) => i.movie.id !== id) })) });
    const r = await api<{ model_version: string }>(`/me/not-interested/${id}`, { method: "PUT" });
    toast("Đã ẩn phim này · gợi ý đã cập nhật", r.model_version);
    reload();
  }

  async function toggleWatchlist(id: number) {
    await api(`/me/watchlist/${id}`, { method: "PUT" });
    toast("Đã thêm vào danh sách xem sau");
  }

  if (error) return <main><div className="empty">Không tải được gợi ý: {error.message}</div></main>;
  if (loading && !data) return <main><div className="skeleton" style={{ height: 340 }} /></main>;
  if (!data) return null;

  const hero = data.hero;
  const reasonFor = (it: RecItem) => (it.reason ? <>Vì bạn đã xem <b>{it.reason.title}</b></> : null);
  const art = data.rows[0]?.items.slice(0, 6) ?? [];

  return (
    <main>
      {data.cold && (
        <div className="banner">
          <span aria-hidden>ℹ️</span>
          <span>
            Bạn mới có <b>{data.n_interactions}</b> tương tác. Chấm thêm <b>{data.threshold - data.n_interactions}</b> phim
            để nhận gợi ý cá nhân hoá — hiện đang hiển thị phim thịnh hành theo thể loại bạn chọn.
          </span>
          <Link className="btn" style={{ marginLeft: "auto", padding: "6px 12px" }} href="/onboarding">Chấm thêm</Link>
        </div>
      )}

      {hero && (
        <section className="hero">
          <div>
            <div className="eyebrow">{data.cold ? "Thịnh hành trong thể loại bạn chọn" : "Gợi ý hàng đầu cho bạn"}</div>
            <h1>{hero.movie.title}</h1>
            <div className="muted">
              {[hero.movie.year, hero.movie.genres.map(gvi).join(" · "), hero.movie.runtime && `${hero.movie.runtime} phút`]
                .filter(Boolean).join(" · ")}
              {hero.movie.rating_count > 0 && <> · <span style={{ color: "var(--accent)" }}>★ {hero.movie.rating_avg.toFixed(2)}</span></>}
            </div>
            {hero.movie.overview && <p className="desc">{hero.movie.overview}</p>}
            {hero.reason && (
              <div className="why">
                Vì bạn đã {hero.reason.rating ? <>chấm <b style={{ color: "var(--text)" }}>{hero.reason.title}</b> {hero.reason.rating} sao</>
                  : <>xem <b style={{ color: "var(--text)" }}>{hero.reason.title}</b></>}
                {hero.reason.shared_genres?.length ? <> — cùng {hero.reason.shared_genres.map(gvi).join(", ")}</> : null}.
              </div>
            )}
            <div className="actions">
              <Link className="btn primary" href={`/movies/${hero.movie.id}`}
                onClick={() => rememberSource(hero.movie.id, { row: "hero", position: 0, request_id: data.request_id, model_version: data.model_version })}>
                ▶ Xem trailer
              </Link>
              <button className="btn" onClick={() => toggleWatchlist(hero.movie.id)}>+ Xem sau</button>
              <button className="btn ghost" onClick={() => hide(hero.movie.id)}>Không quan tâm</button>
            </div>
          </div>
          <div className="art" aria-hidden>
            {art.map((it) => <Poster key={it.movie.id} movie={it.movie} />)}
          </div>
        </section>
      )}

      {data.rows.map((r) => (
        <Row key={r.key} rowKey={r.key} title={rowTitle(r.title, r.kind)} subtitle={r.subtitle} items={r.items}
          ranked={r.kind === "trending"} requestId={data.request_id} modelVersion={data.model_version}
          moreHref={rowMore(r.kind, r.title)} onHide={hide}
          reasonOf={r.kind === "for_you" ? reasonFor : r.kind === "new" ? (it) => genreLine(it.movie) : undefined} />
      ))}

      <p className="dim" style={{ marginTop: 40, fontSize: 12 }}>
        Danh sách sinh bởi <span className="pill-ver">{data.model_version}</span>
      </p>
    </main>
  );
}

export default function HomePage() {
  return <Guard><Home /></Guard>;
}
