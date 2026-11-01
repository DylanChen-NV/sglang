"""Optional early native PD token publication; compute readiness is unchanged."""

import copy
import logging
import os
import time
from array import array

logger = logging.getLogger(__name__)


def enabled():
    value = os.environ.get("SGLANG_Q35RT_EARLY_TOKEN", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_EARLY_TOKEN must be 0 or 1")
    return value == "1" and os.environ.get("SGLANG_Q35RT_PD", "0") == "1"


def pre_draft_enabled():
    value = os.environ.get("SGLANG_Q35RT_PRE_DRAFT_TOKEN", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_PRE_DRAFT_TOKEN must be 0 or 1")
    if value == "1" and not enabled():
        raise ValueError(
            "Pre-draft token publication requires native early token/quiescence"
        )
    return value == "1"


def capture_target_sample(worker, batch, result):
    """Copy only the sampled token before later draft work enters its stream."""
    args = worker.model_runner.server_args
    if (
        not pre_draft_enabled()
        or not args.enable_qwen35_cp_prefill
        or not args.disable_overlap_schedule
        or args.chunked_prefill_size != -1
        or len(batch.reqs) != 1
        or not batch.reqs[0].stream
    ):
        return
    import torch

    token = result.next_token_ids
    if not torch.is_tensor(token) or not token.is_cuda or token.numel() != 1:
        return
    stream = getattr(worker, "_q35rt_sample_copy_stream", None)
    if stream is None:
        stream = worker._q35rt_sample_copy_stream = torch.cuda.Stream(
            device=token.device
        )
    target_done = torch.cuda.Event()
    target_done.record(torch.cuda.current_stream(token.device))
    host = torch.empty(token.shape, dtype=token.dtype, device="cpu", pin_memory=True)
    copied = torch.cuda.Event()
    with torch.cuda.stream(stream):
        stream.wait_event(target_done)
        host.copy_(token, non_blocking=True)
        token.record_stream(stream)
        copied.record(stream)
    result.q35rt_target_sample = (host, copied)


def publish_before_draft(scheduler, batch, result):
    """Publish an output view; full copy_done and native cache processing remain."""
    sample = result.q35rt_target_sample
    if (
        sample is None
        or not pre_draft_enabled()
        or len(batch.reqs) != 1
        or scheduler.server_args.tokenizer_worker_num != 1
    ):
        return False
    req = batch.reqs[0]
    if req.output_ids or req.inflight_middle_chunks > 0 or req.finished_output:
        return False
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.prefill import should_force_retry
    from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group

    if req.pending_bootstrap:
        if should_force_retry(req):
            return False
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            scheduler.attn_cp_cpu_group,
            scheduler.attn_tp_cpu_group,
        )
        # Do not run bootstrap failure/retry cleanup before full GPU completion.
        if polls[0] != KVPoll.WaitingForInput:
            return False
        if not scheduler.disagg_prefill_bootstrap_queue.finalize_bootstrap(req):
            return False
    host, copied = sample
    copied.synchronize()
    token = int(host.reshape(-1)[0])
    if not eligible(req, token, prefill=True):
        return False
    manager = req.disagg_kv_sender.kv_mgr
    if not getattr(manager, "q35rt_quiescence_version", 0):
        raise RuntimeError("Pre-draft token requires negotiated source quiescence")
    view = copy.copy(req)
    view.output_ids = array("q", [token])
    scheduler.disagg_metadata_buffers.set_buf(view)
    req.q35rt_pre_draft_aux_done = manager.q35rt_send_early_aux(
        req.bootstrap_room, req.metadata_buffer_index
    )
    complete_prefill_http(scheduler, req, sampled_token=token)
    logger.info(
        "Q35RT_PRE_DRAFT_TOKEN rid=%s room=%s full_ready=%s ns=%s",
        req.rid,
        req.bootstrap_room,
        int(result.copy_done.query()),
        time.monotonic_ns(),
    )
    return True


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


def complete_prefill_http(scheduler, req, *, sampled_token=None):
    """Close only the P HTTP response; native Req still owns its live transfer."""
    token = (
        sampled_token
        if sampled_token is not None
        else (req.output_ids[0] if len(req.output_ids) == 1 else None)
    )
    if (
        not enabled()
        or not scheduler.server_args.enable_qwen35_cp_prefill
        or scheduler.server_args.tokenizer_worker_num != 1
        or req.finished_output
        or req.pending_bootstrap
        or token is None
        # Native PrefillBootstrapQueue sets its internal output budget to one.
        # Only D needs the client's budget>1 gate; P always returns one token.
        or not eligible(req, token, prefill=True)
    ):
        return False
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    # The output streamer mutates finish/output bookkeeping. Use a shallow
    # response view, so the actual Req remains unfinished in the native inflight
    # queue until every transfer has completed and native cleanup releases it.
    response = copy.copy(req)
    response.output_ids = array("q", [token])
    response.time_stats = copy.copy(req.time_stats)
    response.finished_reason = FINISH_LENGTH(length=0)
    response.finished_output = False
    scheduler.output_streamer.stream_output([response], False)
    # Suppress the later duplicate response; this flag does not release KV,
    # native metadata, Mamba slots, or the sender.
    req.finished_output = True
    return True


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
    if status is None or not (
        status.received_aux or getattr(status, "received_early_aux", False)
    ):
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
