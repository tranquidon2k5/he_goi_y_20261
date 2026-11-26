"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { useAuth } from "./providers";

export function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(-2).map((w) => w[0]?.toUpperCase()).join("") || "?";
}

export function Header() {
  const { user } = useAuth();
  const path = usePathname();
  const router = useRouter();
  const [q, setQ] = useState("");
  if (!user || path === "/onboarding" || path === "/login" || path === "/register") return null;

  const nav = [
    { href: "/", label: "Trang chủ" },
    { href: "/search", label: "Khám phá" },
    { href: "/profile", label: "Danh sách của tôi" },
    ...(user.is_admin ? [{ href: "/admin", label: "Quản trị" }, { href: "/lab", label: "Lab" }] : []),
  ];
  const on = (h: string) => (h === "/" ? path === "/" : path.startsWith(h));

  return (
    <header className="app">
      <Link href="/" className="logo">Cine<span>Rec</span></Link>
      <nav className="main">
        {nav.map((n) => (
          <Link key={n.href} href={n.href} className={on(n.href) ? "on" : ""}>{n.label}</Link>
        ))}
      </nav>
      <form
        className="hsearch"
        role="search"
        onSubmit={(e) => {
          e.preventDefault();
          router.push(`/search?q=${encodeURIComponent(q.trim())}`);
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Tìm phim theo tên…" aria-label="Tìm phim" />
      </form>
      <Link href="/profile" className="avatar" title="Hồ sơ">{initials(user.display_name)}</Link>
    </header>
  );
}
