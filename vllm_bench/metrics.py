"""Read vLLM speculative-decoding counters after a run."""

from vllm.v1.metrics.reader import Counter, Vector


def spec_decode_stats(llm, num_spec_tokens: int) -> dict:
    num_drafts = num_draft_tokens = num_accepted = 0
    acceptance_by_pos = [0] * num_spec_tokens

    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(metric, Counter):
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(metric, Counter):
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(metric, Counter):
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(metric, Vector):
            for pos, val in enumerate(metric.values):
                acceptance_by_pos[pos] += val

    mean_acceptance_length = 1 + (num_accepted / num_drafts) if num_drafts else 1.0
    draft_accept_rate = num_accepted / num_draft_tokens if num_draft_tokens else 0.0

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "mean_acceptance_length": mean_acceptance_length,
        "draft_accept_rate": draft_accept_rate,
        "acceptance_by_pos": acceptance_by_pos,
    }


def print_spec_decode_report(stats: dict) -> None:
    print("\n--- Speculative decoding stats ---")
    print(f"  drafts:           {stats['num_drafts']}")
    print(f"  draft tokens:     {stats['num_draft_tokens']}")
    print(f"  accepted tokens:  {stats['num_accepted_tokens']}")
    print(f"  mean accept len:  {stats['mean_acceptance_length']:.2f}  (1 + accepted/drafts)")
    print(f"  draft accept %:   {stats['draft_accept_rate'] * 100:.1f}%")
    for i, count in enumerate(stats["acceptance_by_pos"]):
        rate = count / stats["num_drafts"] if stats["num_drafts"] else 0.0
        print(f"  pos {i} accept rate: {rate * 100:.1f}%")
