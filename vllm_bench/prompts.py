"""Mixed inference workload — varied length, temperature, and output budget."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class BenchmarkRequest:
    prompt: str
    temperature: float = config.TEMPERATURE
    max_tokens: int = config.MAX_OUTPUT_TOKENS
    category: str = "general"


# Heterogeneous workload: short prompts, long analytical, high-temperature creative.
WORKLOAD: list[BenchmarkRequest] = [
    # --- short prompts (draft overhead dominates; scheduler should skip spec) ---
    BenchmarkRequest("Hi", temperature=0.0, max_tokens=32, category="short"),
    BenchmarkRequest("2+2?", temperature=0.0, max_tokens=16, category="short"),
    BenchmarkRequest("Capital of France?", temperature=0.0, max_tokens=16, category="short"),
    BenchmarkRequest("Define HTTP in one sentence.", temperature=0.0, max_tokens=48, category="short"),
    BenchmarkRequest("Say hello in Japanese.", temperature=0.0, max_tokens=24, category="short"),
    BenchmarkRequest("What is a logit?", temperature=0.0, max_tokens=64, category="short"),
    BenchmarkRequest("JSON or YAML for configs?", temperature=0.0, max_tokens=64, category="short"),
    BenchmarkRequest("Name three sorting algorithms.", temperature=0.0, max_tokens=64, category="short"),
    # --- high temperature (hurts draft acceptance; scheduler should skip spec) ---
    BenchmarkRequest(
        "Write a surreal poem where GPUs dream in binary and wake up as tensors.",
        temperature=1.0,
        max_tokens=200,
        category="high_temp",
    ),
    BenchmarkRequest(
        "Invent a bizarre sci-fi planet with impossible physics. Be wildly creative.",
        temperature=0.95,
        max_tokens=256,
        category="high_temp",
    ),
    BenchmarkRequest(
        "Tell me a random story about a cat who becomes CEO of a startup.",
        temperature=0.9,
        max_tokens=200,
        category="high_temp",
    ),
    BenchmarkRequest(
        "Brainstorm ten absurd product names for an AI-powered toaster.",
        temperature=0.85,
        max_tokens=180,
        category="high_temp",
    ),
    BenchmarkRequest(
        "Describe a dream sequence in the style of magical realism.",
        temperature=1.0,
        max_tokens=220,
        category="high_temp",
    ),
    # --- long, low-temperature (ideal for EAGLE3; scheduler should enable spec) ---
    BenchmarkRequest(
        "Explain how a transformer attention block works in 3 short paragraphs.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Write a Python function that merges two sorted lists in O(n) time with a full explanation.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "What are the main trade-offs between microservices and a monolith for an ML platform?",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Compare FP16, BF16, and FP8 for LLM inference on NVIDIA GPUs. Include numerical range and throughput.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "List three ways to reduce KV-cache memory during long-context inference and when each applies.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Give a 5-step checklist for debugging a slow REST API in production.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Summarize speculative decoding (draft-verify) and why acceptance rate matters for speedup.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Draft a concise email declining a meeting but proposing two alternate times next week.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "Explain tensor parallelism vs pipeline parallelism for serving a 70B model.",
        temperature=0.0,
        category="long_deterministic",
    ),
    BenchmarkRequest(
        "What is continuous batching in vLLM and how does it improve GPU utilization?",
        temperature=0.0,
        category="long_deterministic",
    ),
    # --- medium prompts, moderate temperature (borderline; rolling rate may decide) ---
    BenchmarkRequest(
        "Summarize the plot of The Matrix in under 100 words.",
        temperature=0.3,
        max_tokens=128,
        category="medium",
    ),
    BenchmarkRequest(
        "Outline a blog post on optimizing LLM inference for cost at scale.",
        temperature=0.5,
        max_tokens=256,
        category="medium",
    ),
    BenchmarkRequest(
        "List pros and cons of using LoRA vs full fine-tuning for domain adaptation.",
        temperature=0.4,
        max_tokens=200,
        category="medium",
    ),
    BenchmarkRequest(
        "Explain why batch size affects throughput but not latency per request in many serving stacks.",
        temperature=0.2,
        max_tokens=180,
        category="medium",
    ),
    # --- long output budget (more decode steps → speculation has more room to help) ---
    BenchmarkRequest(
        "Write a detailed tutorial on implementing a simple RAG pipeline with citations.",
        temperature=0.0,
        max_tokens=384,
        category="long_output",
    ),
    BenchmarkRequest(
        "Describe the full lifecycle of an ML model from training to A/B testing in production.",
        temperature=0.0,
        max_tokens=384,
        category="long_output",
    ),
    # --- long context (prompt itself >64 tokens — scheduler should enable spec at temp=0) ---
    BenchmarkRequest(
        "You are reviewing a production LLM serving stack that handles 50k chat requests per day. "
        "The deployment uses vLLM with continuous batching, PagedAttention, tensor parallelism optional, "
        "and EAGLE3 speculative decoding with k=2 draft tokens. Current SLO is p95 latency under 800ms "
        "for 512-token completions at peak concurrency of 32. Baseline decode throughput on one A6000 is "
        "about 120 tokens per second with Llama-3.1-8B at TP=1. Mean draft acceptance length is 1.6. "
        "Explain step by step how you would decide whether to enable speculative decoding for all traffic "
        "versus selectively routing only long, low-temperature requests through the draft path. "
        "List the Prometheus metrics you would monitor during rollout and what thresholds would trigger rollback.",
        temperature=0.0,
        category="long_context",
    ),
    BenchmarkRequest(
        "Context: A fintech company runs batch fraud scoring and real-time user chat on the same GPU cluster. "
        "Fraud prompts are short (under 30 tokens) and need sub-100ms latency. Chat prompts average 400 tokens "
        "with temperature 0.7 for creative support replies. Engineering is considering EAGLE3 speculative "
        "decoding on Llama-3.1-8B to improve chat throughput without buying more GPUs. "
        "Write a structured analysis covering: (1) which request classes benefit from speculation, "
        "(2) which should stay on baseline decode, (3) how prompt length and temperature affect draft acceptance, "
        "(4) a concrete routing policy with numeric thresholds, and (5) how to A/B test before full rollout.",
        temperature=0.0,
        category="long_context",
    ),
    BenchmarkRequest(
        "The following is a truncated architecture doc for an internal inference platform. "
        "Gateway: FastAPI + Redis queue. Workers: vLLM 0.11, one model per GPU, TP=1 for 8B models. "
        "KV cache: FP16, max_model_len=4096, gpu_memory_utilization=0.9. Observability: Prometheus + Grafana, "
        "custom counters for spec_decode_num_drafts and mean acceptance length per model revision. "
        "Pain points: short autocomplete requests see no gain from EAGLE3; high-temperature creative endpoints "
        "show acceptance length below 1.2; long analytical prompts at temperature 0 gain 8-12% throughput. "
        "Given this, propose a selective speculation scheduler with three rules (prompt length, temperature, "
        "rolling accept rate) and explain how each rule prevents wasted draft work in production.",
        temperature=0.0,
        category="long_context",
    ),
    BenchmarkRequest(
        "Compare and contrast three approaches to reducing LLM inference cost at scale: "
        "(A) speculative decoding with EAGLE3 draft models, (B) quantization to FP8 or INT4 weights, "
        "and (C) model distillation to a smaller student. For each approach, discuss throughput impact, "
        "latency impact, memory footprint, operational complexity, and when the technique stops helping. "
        "Assume a baseline of Llama-3.1-8B on a single 48GB GPU with batch size 1-32 and decode-heavy workloads. "
        "Conclude with a recommendation for a mixed chat + code-completion workload where 40% of prompts are "
        "under 50 tokens and 30% use temperature above 0.8.",
        temperature=0.0,
        category="long_context",
    ),
    BenchmarkRequest(
        "A site reliability engineer notices that enabling EAGLE3 globally improved throughput on long "
        "document Q&A by 9% but regressed short-query autocomplete by 4% and had no effect on high-temperature "
        "brainstorming endpoints. Draft acceptance length dropped from 1.7 to 1.3 after a model update. "
        "Describe how you would build a runtime scheduler that routes requests to baseline or speculative engines, "
        "how you would compute rolling acceptance rate from vLLM metrics, and what safe defaults you'd use for "
        "min prompt tokens, max spec temperature, and min rolling accept rate. Include example threshold values "
        "and how you would validate the policy offline before enabling in production.",
        temperature=0.0,
        category="long_context",
    ),
    BenchmarkRequest(
        "Explain the full forward pass of decoder-only transformer inference during autoregressive generation, "
        "including embedding lookup, positional encoding, multi-head self-attention with KV cache reuse, "
        "feed-forward blocks, layer normalization, and final logits projection. Then explain how speculative "
        "decoding changes the loop: a draft model proposes k tokens, the target model verifies them in parallel, "
        "and accepted tokens advance the sequence in one step. Cover why acceptance rate multiplies effective "
        "speedup, why drafting hurts when prompts are very short, and why high sampling temperature reduces "
        "draft-target agreement. Use clear numbered steps suitable for an inference engineering onboarding doc.",
        temperature=0.0,
        category="long_context",
    ),
]

# Quick subset: mix of all categories for smoke tests.
QUICK_WORKLOAD = [
    WORKLOAD[0],   # short
    WORKLOAD[8],   # high temp
    WORKLOAD[29],  # long_context (>64 prompt tokens)
    WORKLOAD[30],  # long_context
    WORKLOAD[24],  # medium
    WORKLOAD[27],  # long output
    WORKLOAD[3],   # short
]
