"use client";

/**
 * Ghi sự kiện tiềm ẩn (PRD mục 4) theo lô: gom trong bộ nhớ, gửi mỗi 5 giây,
 * và gửi bằng sendBeacon khi người dùng rời trang. Server bỏ qua event_id trùng.
 */

type EventType = "impression" | "detail_view" | "watch" | "trailer_play" | "search" | "click";

type Ev = {
  event_id: string;
  type: EventType;
  timestamp: number;
  session_id: string;
  movie_id?: number;
  request_id?: string;
  model_version?: string;
  payload?: Record<string, unknown>;
};

const queue: Ev[] = [];
let timer: ReturnType<typeof setTimeout> | null = null;
let bound = false;

function uid(): string {
  return crypto.randomUUID().replaceAll("-", "");
}

function sessionId(): string {
  try {
    let s = sessionStorage.getItem("cinerec_session");
    if (!s) sessionStorage.setItem("cinerec_session", (s = uid()));
    return s;
  } catch {
    return "nosession";
  }
}

function flush(useBeacon = false) {
  if (timer) clearTimeout(timer);
  timer = null;
  if (!queue.length) return;
  const body = JSON.stringify({ events: queue.splice(0, 500) });
  if (useBeacon && navigator.sendBeacon?.("/api/events", body)) return;
  fetch("/api/events", { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true }).catch(() => {});
}

export function track(type: EventType, data: Omit<Ev, "event_id" | "type" | "timestamp" | "session_id"> = {}) {
  if (typeof window === "undefined") return;
  if (!bound) {
    bound = true;
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && flush(true));
    window.addEventListener("pagehide", () => flush(true));
  }
  queue.push({ event_id: uid(), type, timestamp: Math.floor(Date.now() / 1000), session_id: sessionId(), ...data });
  if (queue.length >= 50) flush();
  else if (!timer) timer = setTimeout(() => flush(), 5000);
}

/** Nguồn click (hàng + vị trí) để gắn vào detail_view ở trang chi tiết. */
export type ClickSource = { row: string; position: number; request_id?: string; model_version?: string };

export function rememberSource(movieId: number, src: ClickSource) {
  try {
    sessionStorage.setItem(`cinerec_src_${movieId}`, JSON.stringify(src));
  } catch {}
}

export function takeSource(movieId: number): ClickSource | null {
  try {
    const k = `cinerec_src_${movieId}`;
    const v = sessionStorage.getItem(k);
    sessionStorage.removeItem(k);
    return v ? JSON.parse(v) : null;
  } catch {
    return null;
  }
}
