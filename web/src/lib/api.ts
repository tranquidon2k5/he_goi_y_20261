"use client";

import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

type Query = Record<string, string | number | boolean | (string | number)[] | null | undefined>;

export function qs(params: Query): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, String(x)));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, ...rest } = init;
  const res = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...rest,
    headers: { ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...rest.headers },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : Array.isArray(body.detail) ? body.detail[0]?.msg ?? msg : msg;
    } catch {}
    throw new ApiError(res.status, msg);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

/** Tải dữ liệu phía client; `reload()` để tải lại sau khi ghi. */
export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [tick, setTick] = useState(0);
  const [doneKey, setDoneKey] = useState<string | null>(null);
  const key = `${path}#${tick}`;

  useEffect(() => {
    if (path === null) return;
    let alive = true;
    api<T>(path)
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e) => alive && setError(e instanceof ApiError ? e : new ApiError(0, String(e))))
      .finally(() => alive && setDoneKey(key));
    return () => {
      alive = false;
    };
  }, [path, key]);

  const loading = path !== null && doneKey !== key;

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload, setData };
}
