import type { Metadata } from "next";
import { Suspense } from "react";
import { Be_Vietnam_Pro } from "next/font/google";
import { Header } from "@/components/header";
import { Providers } from "@/components/providers";
import "./globals.css";

const font = Be_Vietnam_Pro({
  variable: "--font-sans",
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "CineRec",
  description: "Trang xem phim có gợi ý cá nhân hoá — đồ án IT4613",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="vi" className={font.variable}>
      <body>
        <Providers>
          <Suspense fallback={null}>
            <Header />
          </Suspense>
          {children}
        </Providers>
      </body>
    </html>
  );
}
