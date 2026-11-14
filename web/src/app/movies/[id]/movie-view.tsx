"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { genreLine, Row } from "@/components/movie-card";
import { Poster } from "@/components/poster";
import { Guard, useToast } from "@/components/providers";
import { Stars } from "@/components/stars";
import { api, useApi } from "@/lib/api";
import { takeSource, track } from "@/lib/events";
import { gvi } from "@/lib/genres";
import type { MovieDetail, RecItem } from "@/lib/types";

type Similar = { request_id: string; model_version: string; items: RecItem[] };

function Detail({ id }: { id: number }) {
  const { data, error, setData } = useApi<MovieDetail>(`/movies/${id}`);
  const sim = useApi<Similar>(`/movies/${id}/similar`);
  const toast = useToast();
  const [tag, setTag] = useState("");
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const src = takeSource(id);
    track("detail_view", {
      movie_id: id, request_id: src?.request_id, model_version: src?.model_version,
      payload: src ? { source_row: src.row, position: src.position } : {},
    });
  }, [id]);

  if (error) return <main><div className="empty">{error.status === 404 ? "Không tìm thấy phim." : error.message}</div></main>;
  if (!data?.me) return <main><div className="skeleton" style={{ height: 420 }} /></main>;
  const { movie: m, me } = data;
  const patchMe = (p: Partial<NonNullable<MovieDetail["me"]>>) => setData((d) => d && { ...d, me: { ...d.me!, ...p } });

  async function rate(v: number) {
    patchMe({ rating: v, watched: true });
    const r = await api<{ model_version: string }>("/ratings", { method: "POST", json: { movie_id: m.id, rating: v } });
    toast(`Đã chấm ${v} sao · gợi ý đã cập nhật`, r.model_version);
  }
  async function clearRating() {
    patchMe({ rating: null });
    await api(`/ratings/${m.id}`, { method: "DELETE" });
    toast("Đã xoá điểm chấm");
  }
  async function toggle(list: "watchlist" | "not-interested" | "watched", on: boolean, msg: [string, string]) {
    const key = list === "not-interested" ? "not_interested" : list;
    patchMe({ [key]: !on });
    await api(`/me/${list}/${m.id}`, { method: on ? "DELETE" : "PUT" });
    toast(on ? msg[1] : msg[0]);
  }
  async function addTag(t: string) {
    t = t.trim();
    if (!t || me!.tags.some((x) => x.tag === t)) return;
    const r = await api<{ id: number; tag: string }>(`/movies/${m.id}/tags`, { method: "POST", json: { tag: t } });
    patchMe({ tags: [...me!.tags, { id: r.id, tag: r.tag }] });
    setTag("");
    toast(`Đã gắn tag “${t}”`);
  }
  async function removeTag(tid: number) {
    patchMe({ tags: me!.tags.filter((x) => x.id !== tid) });
    await api(`/movies/${m.id}/tags/${tid}`, { method: "DELETE" });
  }
  function play() {
    setPlaying(true);
    track("trailer_play", { movie_id: m.id });
  }

  const facts: [string, React.ReactNode][] = [
    ["Năm", m.year ?? "—"],
    ["Thời lượng", m.runtime ? `${m.runtime} phút` : "—"],
    ["Đạo diễn", m.directors ?? "—"],
    ["Điểm cộng đồng", m.rating_count > 0
      ? <><span style={{ color: "var(--accent)" }}>★ {m.rating_avg.toFixed(2)}</span> <span className="dim">· {m.rating_count.toLocaleString("vi")} lượt</span></>
      : "Chưa có"],
  ];
  const sugg = (data.popular_tags.length ? data.popular_tags : []).filter((t) => !me.tags.some((x) => x.tag === t)).slice(0, 4);

  return (
    <main>
      <Link href="/" className="btn ghost" style={{ marginBottom: 18, padding: "6px 10px" }}>← Trang chủ</Link>
      <div className="detail">
        <div>
          <Poster movie={m} sizes="260px" priority />
          <div className="actions" style={{ flexDirection: "column" }}>
            <button className={`btn ${me.watched ? "on" : ""}`} aria-pressed={me.watched}
              onClick={() => toggle("watched", me.watched, ["Đã đánh dấu đã xem", "Đã bỏ đánh dấu đã xem"])}>
              {me.watched ? "✓ Đã xem" : "◉ Đánh dấu đã xem"}
            </button>
            <button className={`btn ${me.watchlist ? "on" : ""}`} aria-pressed={me.watchlist}
              onClick={() => toggle("watchlist", me.watchlist, ["Đã thêm vào xem sau", "Đã bỏ khỏi xem sau"])}>
              {me.watchlist ? "✓ Trong danh sách xem sau" : "+ Xem sau"}
            </button>
            <button className={`btn ghost ${me.not_interested ? "on" : ""}`} aria-pressed={me.not_interested}
              onClick={() => toggle("not-interested", me.not_interested, ["Đã cập nhật · phim này sẽ không được gợi ý", "Đã cho phép gợi ý lại"])}>
              {me.not_interested ? "Đã ẩn khỏi gợi ý (hoàn tác)" : "⊘ Không quan tâm"}
            </button>
          </div>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {m.genres.map((g) => <span key={g} className="tag">{gvi(g)}</span>)}
            {m.is_new && <span className="tag" style={{ color: "var(--info)", borderColor: "#2b4570" }}>Phim mới</span>}
          </div>
          <h1 style={{ fontSize: 36, marginTop: 10 }}>{m.title}</h1>
          <dl className="facts">
            {facts.map(([k]) => <dt key={k}>{k}</dt>)}
            {facts.map(([k, v]) => <dd key={k}>{v}</dd>)}
          </dl>
          {m.cast && <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>Diễn viên: {m.cast}</p>}
          <p style={{ marginTop: 16, maxWidth: 680, color: "#c9ccd4", fontSize: 15 }}>
            {m.overview || <span className="dim">Chưa có mô tả. Chạy scripts/enrich_tmdb.py để lấy từ TMDB.</span>}
          </p>

          <div className="ratebox">
            <div><div className="dim" style={{ fontSize: 12 }}>Bạn chấm</div><div className="val">{me.rating ? me.rating.toFixed(1) : "–"}</div></div>
            <Stars value={me.rating ?? 0} onChange={rate} />
            <div className="dim" style={{ fontSize: 12, marginLeft: "auto", textAlign: "right" }}>
              0,5–5 sao, bước 0,5<br />
              {me.rating ? <button className="linkbtn" style={{ color: "var(--bad)" }} onClick={clearRating}>Xoá điểm chấm</button> : "Bấm vào sao để chấm"}
            </div>
          </div>

          <div className="taginput">
            <span className="dim" style={{ fontSize: 12 }}>Tag của bạn:</span>
            {me.tags.map((t) => (
              <button key={t.id} className="chip on" style={{ padding: "4px 10px", fontSize: 12 }} title="Bấm để xoá tag"
                onClick={() => removeTag(t.id)}>{t.tag} ✕</button>
            ))}
            <input value={tag} onChange={(e) => setTag(e.target.value)} placeholder="+ thêm tag, Enter" maxLength={100}
              aria-label="Thêm tag" onKeyDown={(e) => e.key === "Enter" && addTag(tag)} />
            {sugg.length > 0 && (
              <span className="dim" style={{ fontSize: 12 }}>
                Gợi ý: {sugg.map((t, i) => <span key={t}>{i > 0 && ", "}<button className="linkbtn" onClick={() => addTag(t)}>{t}</button></span>)}
              </span>
            )}
          </div>

          <div className="trailer">
            {m.trailer_key ? (
              playing ? (
                <iframe src={`https://www.youtube-nocookie.com/embed/${m.trailer_key}?autoplay=1`} title={`Trailer ${m.title}`}
                  allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen />
              ) : (
                <button onClick={play} style={{ width: "100%", height: "100%" }}>
                  <div className="play">▶</div>Xem trailer
                </button>
              )
            ) : (
              <div style={{ textAlign: "center" }}><div className="play">▶</div>Chưa có trailer</div>
            )}
          </div>
        </div>
      </div>

      {sim.data && (
        <Row rowKey="similar" title="Phim tương tự" subtitle="Có thể bạn cũng thích" items={sim.data.items}
          requestId={sim.data.request_id} modelVersion={sim.data.model_version}
          reasonOf={(it) => {
            const shared = it.movie.genres.filter((g) => m.genres.includes(g)).slice(0, 2);
            return shared.length ? shared.map(gvi).join(" · ") : genreLine(it.movie);
          }} />
      )}
    </main>
  );
}

export function MovieView() {
  const { id } = useParams<{ id: string }>();
  return <Guard><Detail key={id} id={Number(id)} /></Guard>;
}
