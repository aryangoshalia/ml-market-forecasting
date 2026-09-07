/** @type {import('next').NextConfig} */
const apiBase = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiBase}/api/:path*` }];
  },
};

export default nextConfig;
