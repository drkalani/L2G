#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.yml}"
COMPOSE_CMD="${COMPOSE_CMD:-}"
SERVICE="${GATEWAY_SERVICE:-l2g-gateway}"

SSL_DOMAIN="${SSL_DOMAIN:-l2g.aiteb.app}"
SSL_DOMAIN_ALIASES="${SSL_DOMAIN_ALIASES:-}"
SSL_EMAIL="${SSL_EMAIL:-ops@${SSL_DOMAIN}}"
SSL_CERTBOT_WEBROOT="${SSL_CERTBOT_WEBROOT:-/var/www/certbot}"
CERTBOT_MODE="${CERTBOT_MODE:-webroot}"
DAYS_BEFORE_RENEW="${DAYS_BEFORE_RENEW:-30}"
RESTART_GATEWAY="${RESTART_GATEWAY:-1}"

if [ -z "$COMPOSE_CMD" ]; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  else
    echo "[ERROR] Docker Compose not found. Install Docker Compose plugin or legacy docker-compose."
    exit 1
  fi
fi

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC2046
  eval "$(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" 2>/dev/null | xargs -d '\n')"
  SSL_DOMAIN="${SSL_DOMAIN:-$SSL_DOMAIN}"
  SSL_DOMAIN_ALIASES="${SSL_DOMAIN_ALIASES:-}"
  SSL_EMAIL="${SSL_EMAIL:-${SSL_EMAIL}}"
  SSL_CERTBOT_WEBROOT="${SSL_CERTBOT_WEBROOT:-$SSL_CERTBOT_WEBROOT}"
  CERTBOT_MODE="${CERTBOT_MODE:-$CERTBOT_MODE}"
fi

COMPOSE_ENV_ARGS=()
if [ -f "$ENV_FILE" ]; then
  COMPOSE_ENV_ARGS+=(--env-file "$ENV_FILE")
fi

log_info() { echo "[INFO] $1"; }
log_success() { echo "[OK] $1"; }
log_warn() { echo "[WARN] $1"; }
log_error() { echo "[ERROR] $1"; }

cert_path() {
  echo "/etc/letsencrypt/live/$SSL_DOMAIN/fullchain.pem"
}

compose() {
  if [ "$COMPOSE_CMD" = "docker compose" ]; then
    docker compose -f "$COMPOSE_FILE" "${COMPOSE_ENV_ARGS[@]}" "$@"
  else
    "$COMPOSE_CMD" -f "$COMPOSE_FILE" "${COMPOSE_ENV_ARGS[@]}" "$@"
  fi
}

compose_services() {
  if [ "$COMPOSE_CMD" = "docker compose" ]; then
    docker compose -f "$COMPOSE_FILE" "${COMPOSE_ENV_ARGS[@]}" config --services
  else
    "$COMPOSE_CMD" -f "$COMPOSE_FILE" "${COMPOSE_ENV_ARGS[@]}" config --services
  fi
}

has_gateway_service() {
  local service="$1"
  compose_services 2>/dev/null | awk 'NF{print $1}' | grep -qx "$service"
}

resolve_gateway_service() {
  if has_gateway_service "$SERVICE"; then
    return 0
  fi

  if [ "$SERVICE" != "gateway" ] && has_gateway_service "gateway"; then
    log_warn "Gateway service '$SERVICE' not found. Falling back to 'gateway'."
    SERVICE="gateway"
    return 0
  fi

  log_error "Gateway service '$SERVICE' not found in $COMPOSE_FILE."
  log_error "Available services:"
  compose_services 2>/dev/null | awk 'NF{print " - "$1}'
  return 1
}

start_gateway() {
  if [ "$RESTART_GATEWAY" = "1" ]; then
    resolve_gateway_service || return
    log_info "Starting $SERVICE..."
    compose up -d "$SERVICE"
  fi
}

refresh_gateway() {
  if [ "$RESTART_GATEWAY" = "1" ]; then
    resolve_gateway_service || return
    log_info "Refreshing $SERVICE config..."
    compose up -d "$SERVICE"
  fi
}

stop_gateway() {
  if [ "$RESTART_GATEWAY" = "1" ]; then
    resolve_gateway_service || return
    log_info "Stopping $SERVICE for certificate operation..."
    compose stop "$SERVICE" >/dev/null 2>&1 || true
  fi
}

with_gateway_cycle() {
  local action="$1"
  if [ "$CERTBOT_MODE" = "standalone" ] && [ "$RESTART_GATEWAY" = "1" ]; then
    stop_gateway
    "$action"
    start_gateway
  else
    "$action"
  fi
}

ensure_tooling() {
  if ! command -v certbot >/dev/null 2>&1; then
    log_error "certbot is not installed."
    exit 1
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    log_error "openssl is required."
    exit 1
  fi
}

build_domain_args() {
  local domains="$SSL_DOMAIN ${SSL_DOMAIN_ALIASES//,/ }"
  local domain
  for domain in $domains; do
    [ -n "$domain" ] && printf '%s\0' "$domain"
  done
}

build_certbot_args() {
  local -a args=()
  local domain
  while IFS= read -r -d '' domain; do
    args+=("-d" "$domain")
  done < <(build_domain_args)
  printf '%s\0' "${args[@]}"
}

certbot_domains_args() {
  local -a args=()
  while IFS= read -r -d '' arg; do
    args+=("$arg")
  done < <(build_certbot_args)
  printf '%s\0' "${args[@]}"
}

days_left() {
  local cert_file
  cert_file="$(cert_path)"
  if [ ! -f "$cert_file" ]; then
    echo -1
    return
  fi

  local expiry epoch_now cert_expiry
  expiry="$(openssl x509 -enddate -noout -in "$cert_file" 2>/dev/null | cut -d= -f2 || true)"
  if [ -z "$expiry" ]; then
    echo -1
    return
  fi

  epoch_now="$(date +%s)"
  cert_expiry="$(date -d "$expiry" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$expiry" +%s 2>/dev/null || echo 0)"
  if [ "$cert_expiry" -le 0 ]; then
    echo -1
    return
  fi
  echo $(( (cert_expiry - epoch_now) / 86400 ))
}

issue_or_renew() {
  local -a certbot_args=()
  while IFS= read -r -d '' arg; do
    certbot_args+=("$arg")
  done < <(certbot_domains_args)

  local plugin_args=()
  if [ "$CERTBOT_MODE" = "webroot" ]; then
    mkdir -p "$SSL_CERTBOT_WEBROOT"
    plugin_args+=("--webroot" "-w" "$SSL_CERTBOT_WEBROOT")
  else
    plugin_args+=("--standalone")
  fi

  certbot certonly \
    "${plugin_args[@]}" \
    --email "$SSL_EMAIL" \
    --agree-tos \
    --no-eff-email \
    --non-interactive \
    --expand \
    --reuse-key \
    "${certbot_args[@]}"
}

renew_now() {
  if [ ! -f "$(cert_path)" ]; then
    log_info "No certificate found for ${SSL_DOMAIN}; issuing a new one..."
    with_gateway_cycle issue_or_renew
    refresh_gateway
    return
  fi

  local renew_cmd=()
  if [ "$CERTBOT_MODE" = "webroot" ]; then
    renew_cmd=(certbot renew --non-interactive --webroot -w "$SSL_CERTBOT_WEBROOT")
  else
    renew_cmd=(certbot renew --non-interactive)
  fi

  if [ "$CERTBOT_MODE" = "standalone" ] && [ "$RESTART_GATEWAY" = "1" ]; then
    stop_gateway
    "${renew_cmd[@]}"
    start_gateway
    refresh_gateway
  else
    "${renew_cmd[@]}"
    refresh_gateway
  fi
}

ssl_status() {
  local days
  days="$(days_left)"
  if [ ! -f "$(cert_path)" ]; then
    log_warn "No certificate found at $(cert_path)."
    return 1
  fi
  if [ "$days" -lt 0 ]; then
    log_warn "Certificate exists but expiry could not be parsed."
    return 1
  fi
  if [ "$days" -lt "$DAYS_BEFORE_RENEW" ]; then
    log_warn "Certificate for ${SSL_DOMAIN} expires in ${days} days."
    return 0
  fi
  log_success "Certificate for ${SSL_DOMAIN} valid for ${days} days."
  return 1
}

ssl_setup_cron() {
  if ! command -v crontab >/dev/null 2>&1; then
    log_error "crontab command not found."
    exit 1
  fi
  (crontab -l 2>/dev/null | grep -v "ssl_manage.sh ssl-auto" || true) | crontab - >/dev/null 2>&1
  (crontab -l 2>/dev/null; echo "0 3,15 * * * cd $ROOT && ./scripts/ssl_manage.sh ssl-auto") | crontab -
  log_success "Cron updated to run ssl-auto at 03:00 and 15:00 daily."
}

show_help() {
  cat <<'EOF'
Usage:
  ./scripts/ssl_manage.sh [command]

Commands:
  ssl-auto        Run if cert is missing or renewal is needed.
  ssl-manual      Issue certificate regardless of current state.
  ssl-renew       Renew existing cert (or issue if missing).
  ssl-status      Show certificate state and days remaining.
  ssl-setup-cron  Create/update cron for ssl-auto.

Environment:
  SSL_DOMAIN             Primary domain (default: l2g.aiteb.app)
  SSL_DOMAIN_ALIASES     Aliases (comma or space separated)
  SSL_EMAIL              ACME contact email
  SSL_CERTBOT_WEBROOT    ACME webroot (default: /var/www/certbot)
  CERTBOT_MODE           webroot (default) or standalone
  DAYS_BEFORE_RENEW      Renewal warning threshold (default: 30)
EOF
}

main() {
  local command="${1:-ssl-auto}"
  ensure_tooling

  case "$command" in
    ssl-auto)
      if ! ssl_status; then
        renew_now
      else
        log_info "Certificate is within allowed window; no action taken."
      fi
      ;;
    ssl-manual)
      with_gateway_cycle issue_or_renew
      refresh_gateway
      ;;
    ssl-renew)
      renew_now
      ;;
    ssl-status)
      ssl_status
      ;;
    ssl-setup-cron)
      ssl_setup_cron
      ;;
    help|-h|--help)
      show_help
      ;;
    *)
      log_error "Unknown command: $command"
      show_help
      exit 1
      ;;
  esac
}

main "$@"
