"""Read SGLang speculative-decoding metrics from /server_info."""

from __future__ import annotations

import requests


def _find_first(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            found = _find_first(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_first(v, key)
            if found is not None:
                return found
    return None


def server_spec_stats(base_url: str) -> dict:
    try:
        info = requests.get(f"{base_url}/server_info", timeout=10).json()
    except (requests.RequestException, ValueError):
        return {}

    accept_len = _find_first(info, "avg_spec_accept_length")
    cur_steps = _find_first(info, "speculative_num_steps")
    stats: dict = {}
    if accept_len is not None:
        stats["mean_acceptance_length"] = float(accept_len)
    if cur_steps is not None:
        stats["current_num_steps"] = int(cur_steps)
    return stats


def print_spec_decode_report(stats: dict) -> None:
    if not stats:
        return
    print("\n--- Speculative decoding stats ---")
    if "mean_acceptance_length" in stats:
        print(f"  mean accept len:    {stats['mean_acceptance_length']:.2f}")
    if "current_num_steps" in stats:
        print(f"  final draft steps:  {stats['current_num_steps']}  (k after adaptation)")
