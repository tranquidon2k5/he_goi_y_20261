"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";
import { useAuth } from "./providers";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const { setUser } = useAuth();
  const router = useRouter();
  const next = useSearchParams().get("next");
  const [form, setForm] = useState({ email: "", password: "", display_name: "" });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const reg = mode === "register";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const u = await api<User>(reg ? "/auth/register" : "/auth/login", {
        method: "POST",
        json: reg ? form : { email: form.email, password: form.password },
      });
      setUser(u);
      router.replace(!u.onboarded ? "/onboarding" : next && next.startsWith("/") ? next : "/");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Có lỗi xảy ra");
      setBusy(false);
    }
  }

  return (
    <main>
      <form className="auth card" onSubmit={submit}>
        <div className="logo" style={{ fontSize: 24 }}>Cine<span>Rec</span></div>
        <h1 style={{ fontSize: 22 }}>{reg ? "Tạo tài khoản" : "Đăng nhập"}</h1>
        {reg && (
          <label>Tên hiển thị
            <input required maxLength={100} value={form.display_name} autoComplete="name"
              onChange={(e) => setForm({ ...form, display_name: e.target.value })} />
          </label>
        )}
        <label>Email
          <input required type="email" value={form.email} autoComplete="email"
            onChange={(e) => setForm({ ...form, email: e.target.value })} />
        </label>
        <label>Mật khẩu
          <input required type="password" minLength={reg ? 6 : undefined} value={form.password}
            autoComplete={reg ? "new-password" : "current-password"}
            onChange={(e) => setForm({ ...form, password: e.target.value })} />
        </label>
        {err && <div className="error" role="alert">{err}</div>}
        <button className="btn primary" disabled={busy} style={{ justifyContent: "center" }}>
          {busy ? "Đang xử lý…" : reg ? "Đăng ký" : "Đăng nhập"}
        </button>
        <div className="muted" style={{ fontSize: 13 }}>
          {reg ? <>Đã có tài khoản? <Link href="/login">Đăng nhập</Link></>
            : <>Chưa có tài khoản? <Link href="/register">Đăng ký</Link></>}
        </div>
        {reg && <p className="dim" style={{ fontSize: 12 }}>
          App chỉ lưu tên hiển thị, email, thể loại bạn chọn và hoạt động trên CineRec. Không hỏi tuổi hay giới tính.
        </p>}
      </form>
    </main>
  );
}
