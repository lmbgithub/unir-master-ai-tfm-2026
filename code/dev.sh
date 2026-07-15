#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f ${SCRIPT_DIR}/docker-compose.yml"

usage() {
  cat <<EOF
Usage: dev.sh <command> [options]

Commands:
  up [--build]          Start all services (add --build to rebuild images)
  down                  Stop all services
  logs [service] [-f]   Show logs (optionally for a specific service, -f to follow)
  migration <args>      Run alembic inside the api container  (e.g. upgrade head)
  db shell              Open a psql shell inside the db container
  db seed               Run the seeder script inside the api container
  db clean              Truncate all tables and re-seed
  restart [--build]     Restart all services (add --build to rebuild images)
EOF
}

cmd="${1:-}"
shift || true

case "$cmd" in
  up)
    if [[ "${1:-}" == "--build" ]]; then
      $COMPOSE up -d --build
    else
      $COMPOSE up -d
    fi
    ;;

  down)
    $COMPOSE down
    ;;

  logs)
    $COMPOSE logs "$@"
    ;;

  migration)
    if [[ $# -eq 0 ]]; then
      echo "Usage: dev.sh migration <alembic args>  (e.g. upgrade head)"
      exit 1
    fi
    $COMPOSE exec api alembic "$@"
    ;;

  restart)
    $COMPOSE down
    if [[ "${1:-}" == "--build" ]]; then
      $COMPOSE up -d --build
    else
      $COMPOSE up -d
    fi
    ;;

  db)
    subcmd="${1:-}"
    case "$subcmd" in
      shell)
        $COMPOSE exec db psql -U urgenurse -d urgenurse
        ;;

      seed)
        $COMPOSE exec api python seeders/seeders.py
        ;;

      clean)
        $COMPOSE exec db psql -U urgenurse -d urgenurse -c \
          "TRUNCATE TABLE case_steps, attachments, cases, users RESTART IDENTITY CASCADE;"
        rm -rf "${SCRIPT_DIR}/../.uploads"/*
        $COMPOSE exec api python seeders/seeders.py
        ;;

      *)
        echo "Unknown db subcommand: '${subcmd}'"
        echo "Available: shell, seed, clean"
        exit 1
        ;;
    esac
    ;;

  *)
    usage
    exit 1
    ;;
esac
