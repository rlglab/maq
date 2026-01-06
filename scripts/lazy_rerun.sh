#!/usr/bin/env bash
# scripts/lazy_rerun.sh
#
# Run from repo root:
#   ./scripts/lazy_rerun.sh
#
# Optional:
#   ./scripts/lazy_rerun.sh -t mytag --auto_evaluate
#   ./scripts/lazy_rerun.sh -trs xxx.pkl -tes yyy.pkl

set -u
set -o pipefail

############################################################
# 0) Resolve root dir (so we can always run train.sh from root)
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_SH="${ROOT_DIR}/scripts/train.sh"
if [ ! -f "$TRAIN_SH" ]; then
  echo "Error: cannot find ${TRAIN_SH}"
  exit 1
fi

############################################################
# 1) Grid
############################################################
ENVS=("door-human-v1" "pen-human-v1" "hammer-human-v1" "relocate-human-v1")
METHODS=("MAQ+RLPD") # "MAQ+DSAC" "MAQ+IQL"
SEQS=(1 2 3 4 5 6 7 8 9)
KS=(16)
SEEDS=(1 10 100)

TAG=""
AUTO_EVAL=true
TRS=""
TES=""

usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -t, --tag                Tag suffix passed to train.sh"
  echo "  -trs, --training_source  Optional training dataset filename (offline_data/...)"
  echo "  -tes, --testing_source   Optional testing dataset filename (offline_data/...)"
  echo "  --auto_evaluate          Pass --auto_evaluate to train.sh"
  echo ""
  echo "Note: envs are fixed in this script:"
  printf "  %s\n" "${ENVS[@]}"
  echo ""
}

while :; do
  case "${1-}" in
    -h|--help) usage; exit 0 ;;
    -t|--tag) shift; TAG="${1-}";;
    -trs|--training_source) shift; TRS="${1-}";;
    -tes|--testing_source) shift; TES="${1-}";;
    --auto_evaluate) AUTO_EVAL=true ;;
    "") break ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
  shift
done

############################################################
# 3) Lockfiles (put on shared fs if multi-machine)
############################################################
LOCK_DIR="${ROOT_DIR}/lockfiles_lazy_rerun"
mkdir -p "$LOCK_DIR"

sanitize() {
  echo "$1" | sed 's/+/_/g' | sed 's/[^A-Za-z0-9_.-]/_/g'
}

############################################################
# 4) Main loop with lockfiles
############################################################
TOTAL=0
SKIPPED=0
FAILED=0

for env in "${ENVS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seqlen in "${SEQS[@]}"; do
      for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          TOTAL=$((TOTAL+1))

          mtag="$(sanitize "$method")"
          lockfile="${LOCK_DIR}/${env}_${mtag}_sq${seqlen}_k${k}_seed${seed}.lock"

          # Atomic lock creation (noclobber)
          if { set -C; 2>/dev/null >"$lockfile"; }; then
            echo "==== [$(hostname)] RUN: env=${env} method=${method} seqlen=${seqlen} k=${k} seed=${seed} ===="

            args=( "--method" "$method"
                   "-env" "$env"
                   "-seqlen" "$seqlen"
                   "-cbsz" "$k"
                   "-s" "$seed"
                 )

            if [ -n "$TAG" ]; then
              args+=( "-t" "$TAG" )
            fi
            if [ -n "$TRS" ]; then
              args+=( "-trs" "$TRS" )
            fi
            if [ -n "$TES" ]; then
              args+=( "-tes" "$TES" )
            fi
            if [ "$AUTO_EVAL" = true ]; then
              args+=( "--auto_evaluate" )
            fi

            # IMPORTANT: run train.sh from ROOT_DIR (train.sh uses relative paths)
            ( cd "$ROOT_DIR" && bash "./scripts/train.sh" "${args[@]}" )
            rc=$?

            if [ $rc -ne 0 ]; then
              echo "!!!! FAILED (rc=$rc): env=${env} method=${method} seqlen=${seqlen} k=${k} seed=${seed}"
              FAILED=$((FAILED+1))
              # Remove lock on failure so it can be retried
              rm -f "$lockfile"
            else
              echo "==== DONE: env=${env} method=${method} seqlen=${seqlen} k=${k} seed=${seed} ===="
            fi

          else
            SKIPPED=$((SKIPPED+1))
            echo "[SKIP] lock exists: ${lockfile}"
          fi
        done
      done
    done
  done
done

echo "=================================================="
echo "lazy_rerun summary:"
echo "  total combos : $TOTAL"
echo "  skipped      : $SKIPPED (lock exists)"
echo "  failed       : $FAILED (lock removed on failure)"
echo "  lock dir     : $LOCK_DIR"
echo "=================================================="
