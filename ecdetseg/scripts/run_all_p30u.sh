#!/usr/bin/env bash
# ARCHIVED entry: new experiments use scripts/run_all.sh (P150u).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export P30U_CONFIG="${P30U_CONFIG:-${P74U_CONFIG:-configs/ecdet/ecdet_s_p30u.yml}}"
export P30U_OUT_ROOT="${P30U_OUT_ROOT:-${P74U_OUT_ROOT:-$HERE/../outputs/p30u}}"
exec bash "$HERE/run_all.sh" "$@"
