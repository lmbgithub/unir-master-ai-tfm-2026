/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // API_URL is a server-side var set in docker-compose to the internal service name.
    // Falls back to localhost for local dev outside Docker.
    const apiUrl = process.env.API_URL ?? "http://localhost:8000";
    console.log("rewrites: API_URL:", apiUrl);
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
  allowedDevOrigins: ["bv.ai"],
};

export default nextConfig;
