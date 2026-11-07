#!/usr/bin/env bash
# Resolve test PID files under the checkpoint; also supports older root-level jobs.
locate_test_pid() {
  local project_dir="$1" run_id="$2" checkpoint_dir
  local -a matches
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || return 2
  if [[ -n "${PID_FILE:-}" ]]; then
    printf '%s\n' "$PID_FILE"
  elif [[ -n "${CKPT:-}" ]]; then
    checkpoint_dir="$(cd "$(dirname "$CKPT")" && pwd)" || return 2
    printf '%s/test_results/%s/%s.pid\n' "$checkpoint_dir" "$run_id" "$run_id"
  elif [[ -f "$project_dir/$run_id.pid" ]]; then
    printf '%s/%s.pid\n' "$project_dir" "$run_id"
  else
    [[ -d "$project_dir/model_pth" ]] || return 2
    mapfile -t matches < <(find "$project_dir/model_pth" -type f -path "*/test_results/$run_id/$run_id.pid")
    [[ "${#matches[@]}" -eq 1 ]] || return 2
    printf '%s\n' "${matches[0]}"
  fi
}
