"use client";

import { useEffect, useState } from "react";
import { Guard, useToast } from "@/components/providers";
import { api, useApi } from "@/lib/api";
import type { LabInfo, ModelVersion } from "@/lib/types";

const KIND: Record<string, string> = { refresh: "làm mới thống kê", finetune: "fine-tune", full: "huấn luyện lại toàn bộ" };
const STATUS: Record<string, string> = { live: "live", pass: "qua cổng", fail: "bị chặn", running: "đang chạy", queued: "chờ chạy", archived: "lưu trữ" };
const METRIC_COLS = ["ndcg@10", "recall@10", "mrr@10", "ndcg@20", "recall@20", "coverage"];
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

  async function retrain() {
    setBusy(true);
    try {
      const v = await api<ModelVersion>("/lab/retrain", { method: "POST", json: { kind: "refresh" } });
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

  const stages = ["Gom sự kiện", "Tính thống kê", "Huấn luyện", "Cổng đánh giá", "Triển khai"];
  const cur = running ? (running.status === "queued" ? 0 : 1) : -1;

  return (
    <main>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <h1>Lab</h1><span className="tag">Chỉ admin · nhóm phát triển & giảng viên</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn" disabled={busy || !!running} onClick={retrain}>↻ Chạy học lại ngay</button>
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
          <div className="d dim">{live?.metrics["ndcg@10"] ? "" : "chưa đánh giá — thuật toán chưa chốt"}</div>
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
          <h2>So sánh offline trên tập test (chia theo thời gian)</h2>
          <span className="dim" style={{ marginLeft: "auto", fontSize: 12 }}>đọc từ model_versions.metrics</span>
        </div>
        {data.offline.length ? (
          <div className="tablewrap">
            <table style={{ marginTop: 12 }}>
              <thead><tr><th>Phiên bản</th><th>Mô hình</th>{METRIC_COLS.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {data.offline.map((r) => (
                  <tr key={r.version}><td className="mono">{r.version}</td><td>{r.recommender}</td>
                    {METRIC_COLS.map((c) => <td key={c} className="mono">{num(r[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty" style={{ marginTop: 12, padding: 36 }}>Chưa có kết quả thí nghiệm offline. Bảng sẽ có dữ liệu khi baseline và mô hình đề xuất được huấn luyện (PRD mục 8–9).</div>}
      </div>

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
