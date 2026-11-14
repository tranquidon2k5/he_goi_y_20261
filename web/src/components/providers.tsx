"use client";

import { usePathname, useRouter } from "next/navigation";
import { createContext, Suspense, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { User } from "@/lib/types";

const PUBLIC = ["/login", "/register"];

type AuthCtx = { user: User | null; ready: boolean; refresh: () => Promise<User | null>; setUser: (u: User | null) => void };
const Auth = createContext<AuthCtx>({ user: null, ready: false, refresh: async () => null, setUser: () => {} });
export const useAuth = () => useContext(Auth);

type ToastCtx = (msg: string, version?: string) => void;
const Toast = createContext<ToastCtx>(() => {});
export const useToast = () => useContext(Toast);

export function Providers({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [toast, setToast] = useState<{ msg: string; version?: string; on: boolean }>({ msg: "", on: false });
  const tt = useRef<ReturnType<typeof setTimeout>>(undefined);

  const refresh = useCallback(async () => {
    try {
      const u = await api<User>("/me");
      setUser(u);
      return u;
    } catch (e) {
      if (!(e instanceof ApiError && e.status === 401)) console.error(e);
      setUser(null);
      return null;
    } finally {
      setReady(true);
    }
  }, []);

  useEffect(() => {
    api<User>("/me")
      .then(setUser, () => setUser(null))
      .finally(() => setReady(true));
  }, []);

  const show = useCallback<ToastCtx>((msg, version) => {
    setToast({ msg, version, on: true });
    clearTimeout(tt.current);
    tt.current = setTimeout(() => setToast((t) => ({ ...t, on: false })), 2600);
  }, []);

  return (
    <Auth.Provider value={{ user, ready, refresh, setUser }}>
      <Toast.Provider value={show}>
        <Suspense fallback={null}><RouteGuard /></Suspense>
        {children}
        <div className={`toast ${toast.on ? "on" : ""}`} role="status" aria-live="polite">
          <span>{toast.msg}</span>
          {toast.version && <span className="ver">model {toast.version}</span>}
        </div>
      </Toast.Provider>
    </Auth.Provider>
  );
}

/** Chỉ render nội dung khi đã có người dùng hợp lệ cho trang này. */
export function Guard({ children, admin = false }: { children: React.ReactNode; admin?: boolean }) {
  const { user, ready } = useAuth();
  if (!ready || !user || (admin && !user.is_admin)) return <div className="skeleton" style={{ height: 320 }} />;
  return <>{children}</>;
}

/** Bảo vệ route: chưa đăng nhập → /login; chưa onboarding → /onboarding; Lab chỉ admin. */
function RouteGuard() {
  const { user, ready } = useAuth();
  const router = useRouter();
  const path = usePathname();
  useEffect(() => {
    if (!ready) return;
    const isPublic = PUBLIC.includes(path);
    if (!user && !isPublic) router.replace(`/login?next=${encodeURIComponent(path)}`);
    else if (user && isPublic) router.replace(user.onboarded ? "/" : "/onboarding");
    else if (user && !user.onboarded && path !== "/onboarding") router.replace("/onboarding");
    else if (user && path.startsWith("/lab") && !user.is_admin) router.replace("/");
  }, [ready, user, path, router]);
  return null;
}
