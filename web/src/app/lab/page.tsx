"use client";

import { useEffect, useState } from "react";
import { Guard, useToast } from "@/components/providers";
import { api, useApi } from "@/lib/api";
import type { Experiments, LabInfo, ModelVersion } from "@/lib/types";

const KIND: Record<string, string> = { refresh: "làm mới thống kê", finetune: "fine-tune", full: "huấn luyện lại toàn bộ" };
const STATUS: Record<string, string> = { live: "live", pass: "qua cổng", fail: "bị chặn", running: "đang chạy", queued: "chờ chạy", archived: "lưu trữ" };
const METRIC_COLS: [string, string][] = [["ndcg@10", "NDCG@10 cổng"], ["test_ndcg@10", "NDCG@10 test"], ["test_recall@10", "Recall@10 test"],
  ["test_mrr", "MRR test"], ["test_coverage@10", "Độ phủ@10"], ["test_diversity@10", "Đa dạng@10"]];
const EXP_COLS = ["ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr", "coverage@10", "diversity@10", "novelty@10"] as const;
const ORDER = ["proposed_v2", "proposed", "esasrec", "gsasrec", "cl4srec", "xsimgcl", "sasrec_ss_content_cl", "sasrec_ss_content", "sasrec_ss", "sasrec"];
const BUCKET: Record<string, string> = { head: "phim phổ biến (20%)", mid: "giữa (30%)", tail: "đuôi dài (50%)" };

function ExperimentsCard({ e }: { e: Experiments }) {
  const names = ORDER.filter((n) => e.models[n]);
  if (!names.length) return null;
  const best = Object.fromEntries(EXP_COLS.map((c) => [c, Math.max(...names.map((n) => e.models[n][c][0]))]));
  const sig = e.significance;
  const f = (x: number, d = 4) => x.toFixed(d);
  const dg = (c: string) => (c === "novelty@10" ? 2 : 4);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <h2>Mô hình đề xuất so với baseline · tập test chia theo thời gian</h2>
        <span className="dim" style={{ marginLeft: "auto", fontSize: 12 }}>
          ML-32M từ {e.data.since_year} · {e.data.users.toLocaleString("vi")} người dùng · {e.data.items.toLocaleString("vi")} phim ·{" "}
          {e.data.interactions.toLocaleString("vi")} tương tác dương
        </span>
      </div>
      <div className="tablewrap">
        <table style={{ marginTop: 12 }}>
          <thead><tr><th>Mô hình</th>{EXP_COLS.map((c) => <th key={c}>{c}</th>)}<th>Seed</th></tr></thead>
          <tbody>
            {names.map((n) => (
              <tr key={n} className={n === "proposed_v2" ? "hl" : ""}>
                <td>{e.labels[n] ?? n}</td>
                {EXP_COLS.map((c) => {
                  const [m, s] = e.models[n][c];
                  return <td key={c} className={`mono ${m === best[c] ? "best" : ""}`}>{f(m, dg(c))}<span className="dim"> ±{f(s, dg(c))}</span></td>;
                })}
                <td className="muted">{e.models[n].seeds}</td>
              </tr>
            ))}
            {Object.entries(e.reference).map(([n, r]) => (
              <tr key={n} className="ref">
                <td className="muted">Tham chiếu: {n}</td>
                {EXP_COLS.map((c) => <td key={c} className="mono muted">{f(r[c], dg(c))}</td>)}
                <td className="muted">—</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {e.significance_v2 && (
        <p style={{ marginTop: 6, fontSize: 13.5 }}>
          Đề xuất v2 so với gSASRec: chênh lệch NDCG@10{" "}
          <b className="mono">{e.significance_v2.mean_diff >= 0 ? "+" : ""}{f(e.significance_v2.mean_diff)}</b>, p ={" "}
          <b className="mono">{e.significance_v2.p < 0.001 ? "< 0.001" : f(e.significance_v2.p, 3)}</b>
          {e.significance_v2.p < 0.05 ? "" : " — không khác có ý nghĩa thống kê."}
        </p>
      )}
      {e.rerank && (
        <div className="tablewrap" style={{ marginTop: 12 }}>
          <h3>Xếp hạng lại theo độ phổ biến gần đây (số ngày, w chọn trên validation)</h3>
          <table className="mono" style={{ marginTop: 6 }}>
            <thead><tr><th>Mô hình</th><th>Số ngày</th><th>w</th><th>val trước → sau</th><th>test trước → sau</th><th>Seed</th></tr></thead>
            <tbody>{names.filter((n) => e.rerank![n]).map((n) => {
              const r = e.rerank![n];
              return <tr key={n} className={n === "proposed_v2" ? "hl" : ""}><td>{e.labels[n] ?? n}</td><td>{r.days}</td><td>{r.w}</td>
                <td>{f(r.val_base)} → {f(r.val_rerank)}</td><td>{f(r.test_base)} → <b>{f(r.test_rerank)}</b></td><td>{r.seeds.length}</td></tr>;
            })}</tbody>
          </table>
        </div>
      )}
      {e.cl_controlled && (
        <p style={{ marginTop: 6, fontSize: 13.5 }}>
          Ablation có kiểm soát — thêm học tương phản vào đúng cấu hình tốt nhất của ablation 3: chênh lệch NDCG@10{" "}
          <b className="mono">{e.cl_controlled.mean_diff >= 0 ? "+" : ""}{f(e.cl_controlled.mean_diff)}</b>, p ={" "}
          <b className="mono">{e.cl_controlled.p < 0.001 ? "< 0.001" : f(e.cl_controlled.p, 3)}</b>.
        </p>
      )}
      {sig && (
        <p style={{ marginTop: 10, fontSize: 13.5 }}>
          Kiểm định t cặp đôi theo người dùng (NDCG@10) giữa mô hình đề xuất và baseline mạnh nhất ({e.labels[sig.vs] ?? sig.vs}):{" "}
          chênh lệch trung bình <b className="mono">{sig.mean_diff >= 0 ? "+" : ""}{f(sig.mean_diff)}</b>, t = {f(sig.t, 2)},{" "}
          p = <b className="mono">{sig.p < 0.001 ? "< 0.001" : f(sig.p, 3)}</b>
          {sig.p < 0.05 ? (sig.mean_diff > 0 ? " — tốt hơn có ý nghĩa thống kê." : " — kém hơn có ý nghĩa thống kê.") : " — khác biệt không có ý nghĩa thống kê (p ≥ 0.05)."}
        </p>
      )}
      <div className="exp-grid">
        <div>
          <h3>Tách theo độ dài lịch sử (NDCG@10)</h3>
          <table className="mono" style={{ marginTop: 6 }}>
            <thead><tr><th>Mô hình</th>{Object.keys(e.models[names[0]].by_history_len).map((b) => <th key={b}>{b}</th>)}</tr></thead>
            <tbody>{names.map((n) => <tr key={n}><td>{n}</td>{Object.values(e.models[n].by_history_len).map((v, i) => <td key={i}>{f(v)}</td>)}</tr>)}</tbody>
          </table>
        </div>
        <div>
          <h3>Tách theo độ phổ biến của phim đích (NDCG@10)</h3>
          <table className="mono" style={{ marginTop: 6 }}>
            <thead><tr><th>Mô hình</th>{Object.keys(e.models[names[0]].by_target_popularity).map((b) => <th key={b} title={BUCKET[b]}>{b}</th>)}</tr></thead>
            <tbody>{names.map((n) => <tr key={n}><td>{n}</td>{Object.values(e.models[n].by_target_popularity).map((v, i) => <td key={i}>{f(v)}</td>)}</tr>)}</tbody>
          </table>
        </div>
        <div>
          <h3>Xếp hạng lại MMR (λ = 0.2)</h3>
          <table className="mono" style={{ marginTop: 6 }}>
            <thead><tr><th>Mô hình</th><th>NDCG@10</th><th>Đa dạng@10</th><th>Độ phủ@10</th></tr></thead>
            <tbody>{names.map((n) => {
              const r = e.models[n]["mmr_0.2"], b = e.models[n];
              return <tr key={n}><td>{n}</td><td>{f(b["ndcg@10"][0])} → {f(r["ndcg@10"])}</td><td>{f(b["diversity@10"][0], 3)} → {f(r["diversity@10"], 3)}</td><td>{f(b["coverage@10"][0], 3)} → {f(r["coverage@10"], 3)}</td></tr>;
            })}</tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
const fmtTime = (t: number | null) => (t ? new Date(t * 1000).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" }) : "—");
const num = (v: unknown) => (typeof v === "number" ? v.toFixed(4) : "—");

function Chart({ versions }: { versions: ModelVersion[] }) {
  const pts = versions.filter((v) => typeof v.metrics["ndcg@10"] === "number").reverse();
  if (!pts.length) {
    return <div className="empty" style={{ marginTop: 18, padding: 36 }}>Chưa có phiên bản nào được chấm NDCG@10. Biểu đồ sẽ hiện khi trainer ghi metric vào model_versions.</div>;
  }
  const vals = pts.map((p) => p.metrics["ndcg@10"]);
  const lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.02;
  return (
    <>
      <div className="chart">
        {pts.map((p) => (
          <div key={p.version} className={`bar ${p.status === "fail" ? "rej" : ""}`} style={{ height: `${((p.metrics["ndcg@10"] - lo) / (hi - lo || 1)) * 100}%` }}>
            <span>{p.metrics["ndcg@10"].toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div className="chart-x">
        {pts.map((p) => <div key={p.version} className="mono">{p.version}<br /><span className="dim">{KIND[p.kind]}</span></div>)}
      </div>
    </>
  );
}

function Lab() {
  const { data, reload } = useApi<LabInfo>("/lab/models");
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const running = data?.versions.find((v) => v.status === "running" || v.status === "queued");

  // theo dõi lần học lại đang chạy
  useEffect(() => {
    if (!running) return;
    const t = setInterval(reload, 1500);
    return () => clearInterval(t);
  }, [running, reload]);

  if (!data) return <main><div className="skeleton" style={{ height: 400 }} /></main>;
  const live = data.live;
  const prev = data.versions.find((v) => v.status === "pass");

  async function retrain(kind: "refresh" | "finetune") {
    setBusy(true);
    try {
      const v = await api<ModelVersion>("/lab/retrain", { method: "POST", json: { kind } });
      toast(`Đã xếp hàng ${v.version} · POST /lab/retrain`);
      reload();
    } catch (e) {
      toast(e instanceof Error ? e.message : "Lỗi");
    } finally {
      setBusy(false);
    }
  }
  async function rollback(version: string) {
    const v = await api<ModelVersion>("/lab/rollback", { method: "POST", json: { version } });
    toast(`Đã rollback · dịch vụ gợi ý nạp lại ${v.version}`, v.version);
    reload();
  }

  const stages = ["Gom sự kiện", "Tính thống kê", "Fine-tune", "Cổng đánh giá", "Triển khai"];
  const cur = running ? (running.status === "queued" ? 0 : 1) : -1;

  return (
    <main>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <h1>Lab</h1><span className="tag">Chỉ admin · nhóm phát triển & giảng viên</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn" disabled={busy || !!running} onClick={() => retrain("refresh")} title="Tính lại độ phổ biến, nạp lại mô hình">↻ Làm mới thống kê</button>
          {live?.recommender === "seqrec" && (
            <button className="btn primary" disabled={busy || !!running} onClick={() => retrain("finetune")}
              title="Fine-tune từ checkpoint đang chạy với dữ liệu mới + replay, qua cổng đánh giá rồi mới triển khai">⚙ Fine-tune ngay</button>
          )}
          {prev && <button className="btn danger" onClick={() => rollback(prev.version)}>⟲ Rollback về {prev.version}</button>}
        </div>
      </div>

      <div className="kpis">
        <div className="card kpi">
          <div className="l">Mô hình đang phục vụ</div>
          <div className="v mono" style={{ fontSize: 20, marginTop: 8 }}>{live?.version ?? "—"}</div>
          <div className="d"><span className="status live"><span className="dot" />live</span> <span className="dim">từ {fmtTime(live?.deployed_at ?? null)}</span></div>
        </div>
        <div className="card kpi">
          <div className="l">NDCG@10 (cửa sổ giữ riêng)</div>
          <div className="v">{num(live?.metrics["ndcg@10"])}</div>
          <div className="d dim">{typeof live?.metrics["ndcg@10"] === "number" ? `${live.metrics["gate_cases"] ?? ""} ca · cửa sổ test MovieLens` : "chưa đánh giá"}</div>
        </div>
        <div className="card kpi">
          <div className="l">Độ trễ API gợi ý p95</div>
          <div className="v">{data.stats.p95_ms !== null ? `${data.stats.p95_ms} ms` : "—"}</div>
          <div className={`d ${data.stats.p95_ms !== null && data.stats.p95_ms > 300 ? "down" : "dim"}`}>ngưỡng 300 ms</div>
        </div>
        <div className="card kpi">
          <div className="l">Tương tác mới chờ học</div>
          <div className="v">{data.stats.pending_interactions.toLocaleString("vi")}</div>
          <div className="d dim">{data.stats.app_users} người dùng app · {data.stats.events.toLocaleString("vi")} sự kiện</div>
        </div>
      </div>

      <div className="lab-grid">
        <div className="card">
          <div style={{ display: "flex", alignItems: "center" }}>
            <h2>NDCG@10 qua các lần học lại</h2>
          </div>
          <Chart versions={data.versions} />
        </div>
        <div className="card">
          <h2>{running ? `Lần học lại đang chạy · ${running.version}` : "Vòng học lại"}</h2>
          <div className="muted" style={{ marginTop: 4, fontSize: 13 }}>
            {running ? `${KIND[running.kind]} từ ${live?.version}` : "Không có lần học lại nào đang chạy."}
          </div>
          <div className="pipeline">
            {stages.map((s, i) => (
              <div key={s} className={`st ${running ? (i < cur ? "done" : i === cur ? "cur" : "") : ""}`}>
                {running && i < cur ? "✓ " : running && i === cur ? "● " : ""}{s}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 20 }}><h3>Cấu hình mô hình đang phục vụ</h3></div>
          <table style={{ marginTop: 6 }} className="mono">
            <tbody>
              <tr><td className="dim">Recommender</td><td>{live?.recommender}</td></tr>
              {Object.entries(live?.config ?? {}).map(([k, v]) => <tr key={k}><td className="dim">{k}</td><td>{JSON.stringify(v)}</td></tr>)}
              <tr><td className="dim">Đã đăng ký</td><td>{data.recommenders.join(", ")}</td></tr>
              {live?.notes && <tr><td className="dim">Ghi chú</td><td style={{ fontFamily: "inherit" }}>{live.notes}</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <h2>Metric của các phiên bản mô hình</h2>
          <span className="dim" style={{ marginLeft: "auto", fontSize: 12 }}>đọc từ model_versions.metrics</span>
        </div>
        {data.offline.length ? (
          <div className="tablewrap">
            <table style={{ marginTop: 12 }}>
              <thead><tr><th>Phiên bản</th><th>Mô hình</th>{METRIC_COLS.map(([c, l]) => <th key={c}>{l}</th>)}</tr></thead>
              <tbody>
                {data.offline.map((r) => (
                  <tr key={r.version}><td className="mono">{r.version}</td><td>{r.recommender}</td>
                    {METRIC_COLS.map(([c]) => <td key={c} className="mono">{num(r[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty" style={{ marginTop: 12, padding: 36 }}>Chưa có phiên bản nào có metric.</div>}
      </div>

      {data.experiments ? <ExperimentsCard e={data.experiments} /> : (
        <div className="card" style={{ marginTop: 16 }}>
          <h2>Mô hình đề xuất so với baseline</h2>
          <div className="empty" style={{ marginTop: 12, padding: 36 }}>Chưa có kết quả thí nghiệm (artifacts/experiments/summary.json do <span className="mono">python -m ml.train report</span> sinh ra).</div>
        </div>
      )}

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Online · CTR theo hàng gợi ý</h2>
        {data.online.length ? (
          <table style={{ marginTop: 12 }}>
            <thead><tr><th>Hàng</th><th>Phim đã hiển thị</th><th>Lượt mở chi tiết</th><th>CTR</th></tr></thead>
            <tbody>
              {data.online.map((o) => (
                <tr key={o.row}><td className="mono">{o.row}</td><td>{o.impressions.toLocaleString("vi")}</td><td>{o.clicks}</td><td className="mono">{(o.ctr * 100).toFixed(1)}%</td></tr>
              ))}
            </tbody>
          </table>
        ) : <p className="dim" style={{ marginTop: 10 }}>Chưa có impression nào.</p>}
        <p className="dim" style={{ fontSize: 12, marginTop: 10 }}>Chỉ mang tính minh hoạ vì ít người dùng (PRD mục 9).</p>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Lịch sử phiên bản (model_versions)</h2>
        <div className="tablewrap">
          <table style={{ marginTop: 12 }}>
            <thead><tr><th>Phiên bản</th><th>Loại</th><th>Trạng thái</th><th>NDCG@10</th><th>Tạo lúc</th><th>Ghi chú</th><th></th></tr></thead>
            <tbody>
              {data.versions.map((v) => (
                <tr key={v.version}>
                  <td className="mono">{v.version}</td>
                  <td>{KIND[v.kind] ?? v.kind}</td>
                  <td><span className={`status ${v.status}`}><span className="dot" />{STATUS[v.status] ?? v.status}</span></td>
                  <td className="mono">{num(v.metrics["ndcg@10"])}</td>
                  <td className="muted">{fmtTime(v.created_at)}</td>
                  <td className="muted" style={{ maxWidth: 320 }}>{v.notes || "—"}</td>
                  <td style={{ textAlign: "right" }}>
                    {v.status === "pass" && <button className="iconbtn" onClick={() => rollback(v.version)}>⟲ Dùng bản này</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}

export default function LabPage() {
  return <Guard admin><Lab /></Guard>;
}
