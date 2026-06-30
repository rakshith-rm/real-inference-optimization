"""Read vLLM speculative-decoding counters after a run."""

from vllm.v1.metrics.reader import Counter, Vector


def spec_decode_stats(llm, num_spec_tokens: int) -> dict:
    num_drafts = num_draft_tokens = num_accepted = 0
    acceptance_by_pos = [0] * num_spec_tokens

    try:
        metrics = llm.get_metrics()
    except (AssertionError, ValueError):
        metrics = []

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(metric, Counter):
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(metric, Counter):
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(metric, Counter):
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(metric, Vector):
            for pos, val in enumerate(metric.values):
                acceptance_by_pos[pos] += val

    return {
        "num_drafts": num_drafts,
        "mean_acceptance_length": 1 + (num_accepted / num_drafts) if num_drafts else 1.0,
        "draft_accept_rate": num_accepted / num_draft_tokens if num_draft_tokens else 0.0,
        "acceptance_by_pos": acceptance_by_pos,
    }


def print_spec_summary(stats: dict | None) -> None:
    if not stats:
        return
    print("\n--- Speculative decoding stats ---")
    print(
        f"  mean accept len: {stats['mean_acceptance_length']:.2f}  "
        f"draft accept %: {stats['draft_accept_rate'] * 100:.1f}"
    )
    for i, c in enumerate(stats["acceptance_by_pos"]):
        rate = c / stats["num_drafts"] if stats["num_drafts"] else 0.0
        print(f"  pos {i} accept rate: {rate * 100:.1f}%")
