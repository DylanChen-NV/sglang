"""Derive R+Q+C+F0+S0+A0 from the validated R+Q A2A harness."""

from pathlib import Path

source = Path("/project/artifacts/dsv4_h20_a2a_rq_benchmark.py").read_text()
replacements = [
    (
        "from sglang.kernels.ops.moe.moe_permute_prepare import moe_permute_prepare",
        "from sglang.kernels.ops.moe.moe_permute_prepare import moe_permute_prepare_with_schedule\n"
        "from sglang.kernels.ops.moe.fused_activation_quant import fused_swiglu_quant_fp8",
    ),
    (
        "def gemm(self,is_fc1,q,scale,offsets,ws):",
        "def gemm(self,is_fc1,q,scale,offsets,counts,schedule,ws,precombined=False):",
    ),
    (
        'return llop.grouped_gemm_out(q,scale,weight,wscale,residual,offsets,ws["cnt"],ws["ts"],ws["te"],ws["tn"],ws["nt"],ws["out"],n,kdim,self.persistent_ctas)',
        'api=llop.grouped_gemm_out_precomputed_schedule_and_scales if precombined else llop.grouped_gemm_out_precomputed_schedule\n'
        '        return api(q,scale,weight,wscale,residual,offsets,counts,ws["ts"],schedule[0],schedule[1],schedule[2],ws["out"],n,kdim,self.persistent_ctas)',
    ),
    (
        "offsets,src2dst=moe_permute_prepare(self.ids,E,is_ep=False)",
        "offsets,src2dst,counts,tile_experts,tile_n,num_tiles=moe_permute_prepare_with_schedule(self.ids,E)\n"
        "        schedule=(tile_experts,tile_n,num_tiles)",
    ),
    (
        "gate_up=self.gemm(True,q1,s1,offsets,self.fc1)",
        "gate_up=self.gemm(True,q1,s1,offsets,counts,schedule,self.fc1)",
    ),
    (
        'self.activated=torch.empty((self.rows,I),dtype=torch.bfloat16,device=DEV)',
        'self.activated=None',
    ),
    (
        'act_and_mul_triton(gate_up,self.activated,{},activation="silu",swiglu_limit=LIMIT)\n'
        '        q2,s2=hops.quant_input(self.activated,"float8e4m3",outputs=self.fc2["q"])',
        'q2,s2=fused_swiglu_quant_fp8(gate_up,offsets,lr2,LIMIT,outputs=self.fc2["q"])',
    ),
    (
        "down=self.gemm(False,q2,s2,offsets,self.fc2)",
        "down=self.gemm(False,q2,s2,offsets,counts,schedule,self.fc2,precombined=True)",
    ),
]

for old, new in replacements:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, got {count}: {old[:80]}")
    source = source.replace(old, new)

exec(compile(source, "dsv4_h20_a2a_a0_variant[derived]", "exec"))
