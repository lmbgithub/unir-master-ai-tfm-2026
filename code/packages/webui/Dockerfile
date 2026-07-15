FROM node:20-alpine AS base

ARG NEXT_PUBLIC_API_URL
ARG API_URL
ENV API_URL=${API_URL}

WORKDIR /app
COPY package*.json ./
RUN npm ci

# ── dev: source mounted via volume; Next.js HMR handles reload ────────────
FROM base AS dev

COPY . .
CMD ["npm", "run", "dev"]

# ── prod ──────────────────────────────────────────────────────────────────
FROM base AS prod

COPY . .
RUN npm run build
CMD ["npm", "start"]
