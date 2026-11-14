"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { initials } from "@/components/header";
import { MovieCard } from "@/components/movie-card";
import { Poster } from "@/components/poster";
import { Guard, useAuth, useToast } from "@/components/providers";
import { Stars } from "@/components/stars";
import { api, useApi } from "@/lib/api";
import { gvi } from "@/lib/genres";
import type { Movie } from "@/lib/types";

type RatingRow = { movie: Movie; rating: number; timestamp: number; source: string; tags: string[] };
type ListRow = { movie: Movie; timestamp: number };
type Tab = "ratings" | "watchlist" | "not-interested" | "account";

const fmt = (t: number) =>
  new Date(t * 1000).toLocaleString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });

function Profile() {
  const { user, setUser, refresh } = useAuth();
  const toast = useToast();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("ratings");
  const ratings = useApi<{ items: RatingRow[] }>("/me/ratings");
  const wl = useApi<{ items: ListRow[] }>("/me/watchlist");
  const ni = useApi<{ items: ListRow[] }>("/me/not-interested");
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(user!.display_name);
  const [confirmDel, setConfirmDel] = useState(false);

  const rs = ratings.data?.items ?? [];
  const avg = rs.length ? (rs.reduce((a, b) => a + b.rating, 0) / rs.length).toFixed(2) : "–";

  async function editRating(id: number, v: number) {
    ratings.setData((d) => d && { items: d.items.map((x) => (x.movie.id === id ? { ...x, rating: v } : x)) });
    await api("/ratings", { method: "POST", json: { movie_id: id, rating: v } });
    toast(`Đã sửa thành ${v} sao`);
  }
  async function delRating(id: number) {
    ratings.setData((d) => d && { items: d.items.filter((x) => x.movie.id !== id) });
    await api(`/ratings/${id}`, { method: "DELETE" });
    toast("Đã xoá điểm chấm · gợi ý đã cập nhật");
    refresh();
  }
  async function removeFrom(list: "watchlist" | "not-interested", id: number) {
    (list === "watchlist" ? wl : ni).setData((d) => d && { items: d.items.filter((x) => x.movie.id !== id) });
    await api(`/me/${list}/${id}`, { method: "DELETE" });
    toast("Đã cập nhật");
  }
  async function saveName() {
    const u = await api<typeof user>("/me", { method: "PATCH", json: { display_name: name } });
    setUser(u);
    setEditing(false);
    toast("Đã lưu hồ sơ");
  }
  async function logout() {
    await api("/auth/logout", { method: "POST" });
    setUser(null);
    router.replace("/login");
  }
  async function deleteAccount() {
    await api("/me", { method: "DELETE" });
    setUser(null);
    router.replace("/register");
  }

  const tabs: [Tab, string][] = [["ratings", "Lịch sử chấm"], ["watchlist", "Xem sau"], ["not-interested", "Không quan tâm"], ["account", "Tài khoản & dữ liệu"]];
  const list = tab === "watchlist" ? wl : ni;

  return (
    <main>
      <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
        <div className="avatar" style={{ width: 64, height: 64, fontSize: 22 }}>{initials(user!.display_name)}</div>
        <div>
          {editing ? (
            <form style={{ display: "flex", gap: 8 }} onSubmit={(e) => { e.preventDefault(); saveName(); }}>
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={100} required aria-label="Tên hiển thị" />
              <button className="btn primary">Lưu</button>
              <button type="button" className="btn ghost" onClick={() => setEditing(false)}>Huỷ</button>
            </form>
          ) : <h1>{user!.display_name}</h1>}
          <div className="muted">
            {user!.email} · Thể loại yêu thích: {user!.genres.map(gvi).join(", ") || "—"}{" "}
            <Link href="/onboarding" className="linkbtn">đổi</Link>
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
          {!editing && <button className="btn" onClick={() => setEditing(true)}>Sửa hồ sơ</button>}
          <button className="btn ghost" onClick={logout}>Đăng xuất</button>
        </div>
      </div>

      <div className="kpis">
        <div className="card kpi"><div className="l">Phim đã chấm</div><div className="v">{rs.length}</div></div>
        <div className="card kpi"><div className="l">Danh sách xem sau</div><div className="v">{wl.data?.items.length ?? 0}</div></div>
        <div className="card kpi"><div className="l">Đã đánh dấu “Không quan tâm”</div><div className="v">{ni.data?.items.length ?? 0}</div></div>
        <div className="card kpi"><div className="l">Điểm trung bình bạn chấm</div><div className="v">{avg}</div></div>
      </div>

      <div className="tabs" role="tablist">
        {tabs.map(([k, l]) => (
          <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
        ))}
      </div>

      {tab === "ratings" && (rs.length ? (
        <>
          <div className="tablewrap">
            <table>
              <thead><tr><th></th><th>Phim</th><th>Điểm của bạn</th><th>Tag</th><th>Thời điểm</th><th></th></tr></thead>
              <tbody>
                {rs.map((r) => (
                  <tr key={r.movie.id}>
                    <td style={{ width: 52 }}><Poster movie={r.movie} className="mini" sizes="36px" /></td>
                    <td>
                      <Link href={`/movies/${r.movie.id}`} style={{ fontWeight: 600, textDecoration: "none" }}>{r.movie.title}</Link>
                      <div className="dim" style={{ fontSize: 12 }}>{r.movie.year} · {r.movie.genres.slice(0, 2).map(gvi).join(", ")}</div>
                    </td>
                    <td style={{ whiteSpace: "nowrap" }}>
                      <Stars small value={r.rating} onChange={(v) => editRating(r.movie.id, v)} />
                      <span className="muted" style={{ marginLeft: 6 }}>{r.rating.toFixed(1)}</span>
                    </td>
                    <td>{r.tags.length ? r.tags.map((t) => <span key={t} className="tag" style={{ marginRight: 4 }}>{t}</span>) : <span className="dim">—</span>}</td>
                    <td className="muted">{fmt(r.timestamp)}</td>
                    <td style={{ textAlign: "right" }}>
                      <button className="iconbtn del" onClick={() => delRating(r.movie.id)} aria-label={`Xoá điểm ${r.movie.title}`}>🗑 Xoá</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="dim" style={{ fontSize: 12, marginTop: 12 }}>Sửa điểm bằng cách bấm vào sao. Gợi ý của bạn sẽ thay đổi theo.</p>
        </>
      ) : <div className="empty">Bạn chưa chấm phim nào.</div>)}

      {(tab === "watchlist" || tab === "not-interested") && (list.data?.items.length ? (
        <div className="result-grid">
          {list.data.items.map((x) => (
            <div key={x.movie.id}>
              <MovieCard movie={x.movie} />
              <button className="btn ghost" style={{ width: "100%", marginTop: 6, padding: 6, justifyContent: "center" }}
                onClick={() => removeFrom(tab, x.movie.id)}>
                {tab === "watchlist" ? "Bỏ khỏi xem sau" : "Cho phép gợi ý lại"}
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty">
          {tab === "watchlist" ? "Danh sách xem sau đang trống. Bấm “+ Xem sau” ở trang phim để thêm." : "Bạn chưa ẩn phim nào."}
        </div>
      ))}

      {tab === "account" && (
        <div className="lab-grid">
          <div className="card">
            <h3>Dữ liệu app lưu về bạn</h3>
            <ul className="muted" style={{ margin: "10px 0 0 18px", lineHeight: 1.9 }}>
              <li>Tên hiển thị, email, thể loại đã chọn</li>
              <li>Điểm chấm và tag của bạn</li>
              <li>Hoạt động trên CineRec: phim bạn mở, trailer đã xem, danh sách xem sau, phim bạn ẩn, từ khoá tìm kiếm</li>
              <li>Không thu thập tuổi, giới tính</li>
            </ul>
            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
              <a className="btn" href="/api/me/export" download>⬇ Tải dữ liệu của tôi (.csv)</a>
            </div>
          </div>
          <div className="card" style={{ borderColor: "#4a2a2a" }}>
            <h3 style={{ color: "var(--bad)" }}>Xoá tài khoản</h3>
            <p className="muted" style={{ marginTop: 8 }}>
              Xoá vĩnh viễn tài khoản cùng toàn bộ điểm chấm, tag và lịch sử hoạt động. Dữ liệu của bạn sẽ không còn được dùng để gợi ý.
            </p>
            {confirmDel ? (
              <div style={{ display: "flex", gap: 10, marginTop: 16, alignItems: "center", flexWrap: "wrap" }}>
                <span className="error">Không thể hoàn tác. Chắc chắn?</span>
                <button className="btn danger" onClick={deleteAccount}>Xoá vĩnh viễn</button>
                <button className="btn ghost" onClick={() => setConfirmDel(false)}>Huỷ</button>
              </div>
            ) : (
              <button className="btn danger" style={{ marginTop: 16 }} onClick={() => setConfirmDel(true)}>Xoá tài khoản và dữ liệu</button>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

export default function ProfilePage() {
  return <Guard><Profile /></Guard>;
}
