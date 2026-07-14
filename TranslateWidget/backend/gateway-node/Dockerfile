FROM node:18-slim

WORKDIR /app

COPY backend/gateway-node/package*.json ./
RUN npm install --omit=dev

COPY backend/gateway-node/ ./
COPY widget/ ../../widget/

EXPOSE 8787

CMD ["node", "server.js"]
