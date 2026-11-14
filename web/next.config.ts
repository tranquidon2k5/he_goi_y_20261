import type { NextConfig } from "next";

const API_URL = process.env.CINEREC_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // Gọi FastAPI qua cùng origin để cookie đăng nhập (httpOnly) đi kèm mọi request
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
  images: {
    remotePatterns: [{ protocol: "https", hostname: "image.tmdb.org", pathname: "/t/p/**" }],
  },
};

export default nextConfig;
