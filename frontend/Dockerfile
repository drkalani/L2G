# syntax=docker/dockerfile:1

FROM node:20 AS build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE=/api
ARG VITE_COLAB_GITHUB_USERNAME=drkalani
ARG VITE_COLAB_GITHUB_REPO=L2G
ARG VITE_COLAB_GITHUB_BRANCH=main
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_COLAB_GITHUB_USERNAME=$VITE_COLAB_GITHUB_USERNAME
ENV VITE_COLAB_GITHUB_REPO=$VITE_COLAB_GITHUB_REPO
ENV VITE_COLAB_GITHUB_BRANCH=$VITE_COLAB_GITHUB_BRANCH
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
