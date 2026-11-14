"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Poster } from "@/components/poster";
import { Guard, useAuth, useToast } from "@/components/providers";
import { Stars } from "@/components/stars";
import { api, useApi } from "@/lib/api";
import { GLYPH, gvi } from "@/lib/genres";
import type { Movie } from "@/lib/types";

const MIN_GENRES = 2;
const MIN_RATINGS = 5;
const PAGE = 20;

type PoolItem = { movie: Movie; my_rating: number | null };

function Onboarding() {
  const { user, refresh } = useAuth();
  const toast = useToast();
  const router = useRouter();
  const [step, setStep] = useState<1 | 2>(user?.onboarded ? 2 : 1);
  const [genres, setGenres] = useState<Set<string>>(new Set(user?.genres));
  const all = useApi<string[]>("/movies/genres");
  const [pool, setPool] = useState<PoolItem[]>([]);
  const [ratings, setRatings] = useState<Map<number, number>>(new Map());
  const [busy, setBusy] = useState(false);

  const fetchPool = (offset: number) => api<{ items: PoolItem[] }>(`/movies/onboarding?offset=${offset}&limit=${PAGE}`);
  function addPool(items: PoolItem[]) {
    setPool((p) => [...p, ...items.filter((x) => !p.some((y) => y.movie.id === x.movie.id))]);
    setRatings((m) => {
      const n = new Map(m);
      items.forEach((x) => x.my_rating && n.set(x.movie.id, x.my_rating));
      return n;
    });
  }
  const loadMore = (offset: number) => fetchPool(offset).then((r) => addPool(r.items));

  // người dùng đã onboarding quay lại để "Chấm thêm" → vào thẳng bước 2
  useEffect(() => {
    if (user?.onboarded) fetchPool(0).then((r) => addPool(r.items));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function toStep2() {
    setBusy(true);
    await api("/me", { method: "PATCH", json: { genres: [...genres] } });
    setBusy(false);
    setPool([]);
    setStep(2);
    await loadMore(0);
  }

  async function rate(id: number, v: number) {
    setRatings((m) => new Map(m).set(id, v));
    try {
      await api("/ratings", { method: "POST", json: { movie_id: id, rating: v } });
    } catch (e) {
      toast(e instanceof Error ? e.message : "Không lưu được điểm");
    }
  }

  async function finish() {
    setBusy(true);
    await api("/me", { method: "PATCH", json: { onboarded: true } });
    await refresh();
    toast("Xong! Đây là gợi ý dành cho bạn");
    router.replace("/");
  }

  const n = ratings.size;
  return (
    <main>
      <div className="ob">
        <div className="steps"><div className="on" /><div className={step === 2 ? "on" : ""} /></div>
        {step === 1 ? (
          <div>
            <div className="muted" style={{ fontSize: 13 }}>Bước 1 / 2</div>
            <h1 style={{ marginTop: 4 }}>Bạn thích xem thể loại nào?</h1>
            <p className="muted" style={{ marginTop: 6 }}>Chọn ít nhất {MIN_GENRES} thể loại để chúng tôi biết nên gợi ý gì cho bạn trước.</p>
            <div className="genre-grid">
              {(all.data ?? []).map((g) => (
                <button key={g} className={`chip ${genres.has(g) ? "on" : ""}`} aria-pressed={genres.has(g)}
                  onClick={() => setGenres((s) => { const n = new Set(s); if (n.has(g)) n.delete(g); else n.add(g); return n; })}>
                  <span aria-hidden>{GLYPH[g] ?? "✦"}</span>{gvi(g)}
                </button>
              ))}
            </div>
            <button className="btn primary" disabled={genres.size < MIN_GENRES || busy} onClick={toStep2}>Tiếp tục →</button>
          </div>
        ) : (
          <div>
            <div className="muted" style={{ fontSize: 13 }}>Bước 2 / 2</div>
            <h1 style={{ marginTop: 4 }}>Chấm vài phim bạn đã xem</h1>
            <p className="muted" style={{ marginTop: 6 }}>
              Chấm tối thiểu {MIN_RATINGS} phim (0,5–5 sao). Chưa xem thì bỏ qua. Càng chấm nhiều, gợi ý càng sát.
            </p>
            <div className="rate-grid">
              {pool.map(({ movie }) => {
                const r = ratings.get(movie.id) ?? 0;
                return (
                  <div key={movie.id} className={`item ${r ? "done" : ""}`}>
                    <Poster movie={movie} />
                    {movie.poster_url && <div style={{ fontSize: 12, fontWeight: 600, textAlign: "center" }}>{movie.title}</div>}
                    <Stars small value={r} onChange={(v) => rate(movie.id, v)} />
                    <span className="dim" style={{ fontSize: 11 }}>{r ? `${r} sao` : "Chưa xem? Bỏ qua"}</span>
                  </div>
                );
              })}
            </div>
            <div className="ob-foot">
              <div className="counter">Đã chấm <em>{n}</em> / {MIN_RATINGS}</div>
              <div className="muted" style={{ fontSize: 13 }}>
                {n >= MIN_RATINGS ? "Đủ rồi! Bạn có thể chấm thêm hoặc tiếp tục." : `Còn ${MIN_RATINGS - n} phim nữa để mở “Dành cho bạn”`}
              </div>
              <button className="btn ghost" style={{ marginLeft: "auto" }} onClick={() => setStep(1)}>← Thể loại</button>
              <button className="btn ghost" onClick={() => loadMore(pool.length)}>Hiện phim khác</button>
              <button className="btn primary" disabled={n < MIN_RATINGS || busy} onClick={finish}>Xem gợi ý của tôi</button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default function OnboardingPage() {
  return <Guard><Onboarding /></Guard>;
}
