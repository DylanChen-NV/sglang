"""Optional early native PD token publication; compute readiness is unchanged."""

import copy
import os


def enabled():
    value = os.environ.get("SGLANG_Q35RT_EARLY_TOKEN", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_EARLY_TOKEN must be 0 or 1")
    return value == "1" and os.environ.get("SGLANG_Q35RT_PD", "0") == "1"


def eligible(req, token, *, prefill=False):
    """Leave finish-sensitive and constrained requests on native completion."""
    params = req.sampling_params
    if (
        not req.stream
        or req.finished()
        or req.to_finish is not None
        or (not prefill and params.max_new_tokens <= 1)
        or params.stop_strs
        or params.stop_regex_strs
        or req.grammar_key is not None
        or req.grammar is not None
        or req.session is not None
        or req.return_logprob
        or req.return_hidden_states
        or req.return_routed_experts
        or req.return_indexer_topk
        or req.return_sampling_mask
    ):
        return False
    if not params.ignore_eos:
        if token in (params.stop_token_ids or ()) or token in (req.eos_token_ids or ()):
            return False
        if req.tokenizer is not None:
            if token == req.tokenizer.eos_token_id:
                return False
            if token in (req.tokenizer.additional_stop_token_ids or ()):
                return False
    return True


def complete_prefill_http(scheduler, req):
    """Close only the P HTTP response; native Req still owns its live transfer."""
    if (
        not enabled()
        or not scheduler.server_args.enable_qwen35_cp_prefill
        or scheduler.server_args.tokenizer_worker_num != 1
        or req.pending_bootstrap
        or len(req.output_ids) != 1
        # Native PrefillBootstrapQueue sets its internal output budget to one.
        # Only D needs the client's budget>1 gate; P always returns one token.
        or not eligible(req, req.output_ids[0], prefill=True)
    ):
        return
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    # The output streamer mutates finish/output bookkeeping. Use a shallow
    # response view, so the actual Req remains unfinished in the native inflight
    # queue until every transfer has completed and native cleanup releases it.
    response = copy.copy(req)
    response.time_stats = copy.copy(req.time_stats)
    response.finished_reason = FINISH_LENGTH(length=0)
    response.finished_output = False
    scheduler.output_streamer.stream_output([response], False)
    # Suppress the later duplicate response; this flag does not release KV,
    # native metadata, Mamba slots, or the sender.
    req.finished_output = True


def emit_decode_token(queue, decode_req):
    """Publish a metadata-ready token without admitting any decode GPU work."""
    req = decode_req.req
    if not enabled() or decode_req.is_rebootstrap or req.output_ids:
        return False
    receiver = decode_req.kv_receiver
    quiet = getattr(receiver, "_q35rt_quiet", None)
    if quiet is not None and quiet.pending:
        return False
    manager = receiver.kv_mgr
    if not hasattr(manager, "transfer_statuses"):
        return False
    status = manager.transfer_statuses.get(req.bootstrap_room)
    if status is None or not status.received_aux:
        return False
    buffers = queue.metadata_buffers.get_buf(decode_req.metadata_buffer_index)
    # Native aux completion fences the CPU buffers. The room guard prevents a
    # stale slot from ever publishing another request's token.
    if int(buffers[-1][0]) != req.bootstrap_room:
        return False
    token = int(buffers[0][0])
    if not eligible(req, token):
        return False
    pool = queue.scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not hasattr(pool, "q35rt_import_received"):
        return False
    req.output_ids.append(token)
    req.q35rt_early_token = token
    cached = buffers[1]
    delta = max(0, len(req.prefix_indices) - int(cached[0]))
    req.cached_tokens = int(cached[0]) + delta
    req.cached_tokens_device = int(cached[1]) + delta
    req.cached_tokens_host = int(cached[2])
    req.cached_tokens_storage = int(cached[3])
    queue.scheduler.output_streamer.stream_output([req], False)
    return True
