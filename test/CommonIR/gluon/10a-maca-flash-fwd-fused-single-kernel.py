"""C500/MC550 FlashAttention forward - 13g: 13d + per-stage iglp windows.

Based on 13d (3-stage pipeline), adds iglp after each major computation stage
following the matmul reference pattern (work → iglp → barrier):

  store K/V          → iglp(sts config)     → barrier
  QK[i]              → iglp(tensor config)        ← Tensor window
  PV[i-2]            → iglp(tensor config)        ← Tensor window
  softmax[i-1]       → iglp(vector config)  → barrier

Per beat: 4 iglp calls. Per pair (i0/i1): 8 calls.

Original 13d description:

    beat i:  store K[i], V[i-2] -> smem (single bank)
             barrier
             QK[i]              -> S[i]        (MMA, newest)
             prefetch K[i+1], V[i-1] -> regs   (V delayed by two beats)
             softmax(S[i-1])    -> P[i-1]      (ALU, middle)
             PV: P[i-2] @ V[i-2]              (MMA, oldest — uses P bank!)
             barrier

Key difference from 13c: PV uses P[i-2] (computed 2 beats ago, held in P bank)
instead of P[i-1] (computed in the same beat).  This breaks the serial
dependency softmax -> PV within a beat:

  13c:  QK[i](Tensor) | softmax[i-1](Vector) | PV[i-1](Tensor)  ← PV waits for softmax
  13d:  QK[i](Tensor) + PV[i-2](Tensor) | softmax[i-1](Vector)  ← Tensor continuous

P banks are double-buffered (p_bank_a / p_bank_b, like S banks).
V lifecycle: prefetch at beat i (V[i-1]), store at beat i+1, PV at beat i+1.

Register costs vs 13c:
  + P bank (2 × [64,32] fp16) — softmax result survives 1 extra beat
  + V[i-1] stays in regs one extra beat (prefetch delay 2 vs 1)
  S bank unchanged (already double-buffered).

Scope: ONLY the (BLOCK_M==64, CAUSAL, EVEN_MN) dense region is pipelined.
All other paths keep the original _flash_fwd_tile loop verbatim.

Correctness guards:
  * S banks init to -inf  -> early softmax produces P=0
  * P banks init to 0     -> early PV multiplies zeros
  * V[-2], V[-1] = 0     -> early PV multiplies zeros
  * out-of-range K/V masked to 0, scores poisoned to -inf
  * 2 extra drain beats (dense_n+2 total) flush the pipeline
"""

import argparse
import math

import torch
import triton
from flash_attn.flash_attn_interface import _flash_attn_forward

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from maca_compose_shared_layout import (
    compose_maca_shared_layout_for_operand,
)


BM64 = 64
BM128 = 128
BN = 64
D = 128
NUM_WARPS = 4
HEADS = 32
LOG2E = 1.4426950408889634
TEST_SEQ_LENS = (1024, 4096, 10240)


@gluon.jit
def _flash_fwd_tile(
    n_block,
    k_register,
    acc_o,
    m_i,
    l_i,
    q_for_score,
    q_m,
    k_head,
    v_head,
    k_stride_n,
    k_stride_d,
    v_stride_n,
    v_stride_d,
    SQ,
    SK,
    scale_log2,
    first_d,
    k_tn_smem,
    v_fragments,
    tn_layout: gl.constexpr,
    global_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    MASKED: gl.constexpr,
    CAUSAL: gl.constexpr,
    qk_mma_layout: gl.constexpr,
    qk_b_layout: gl.constexpr,
    pv_mma_layout: gl.constexpr,
    pv_a_layout: gl.constexpr,
    pv_b_layout: gl.constexpr,
):
    # Unchanged from 10-*: per-tile stage used by masked tiles and all
    # non-pipelined paths.
    k_tn_fragments = k_tn_smem._reinterpret(
        k_head.dtype.element_ty,
        [BLOCK_N // 32, BLOCK_D, 32],
        tn_layout,
    )
    current_n = n_block * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=gl.SliceLayout(1, global_layout),
    )
    n0 = n_block * BLOCK_N + gl.arange(0, 32)
    n1 = n_block * BLOCK_N + 32 + gl.arange(0, 32)

    k0_smem = k_tn_fragments.index(0)
    k1_smem = k_tn_fragments.index(1)
    if BLOCK_M == 64:
        k0_smem.store(
            gl.metax.slice(k_register, [32, BLOCK_D], [0, 0]).permute(1, 0)
        )
        k1_smem.store(
            gl.metax.slice(k_register, [32, BLOCK_D], [32, 0]).permute(1, 0)
        )
    else:
        k_tn_smem.store(k_register.permute(1, 0))
    current_v_ptrs = (
        v_head
        + current_n[:, None] * v_stride_n
        + first_d[None, :] * v_stride_d
    )
    if MASKED:
        v_register = gl.load(
            current_v_ptrs,
            mask=current_n[:, None] < SK,
            other=0.0,
        )
    else:
        v_register = gl.load(current_v_ptrs)

    gl.metax.barrier_shared()
    kt0_for_score = k0_smem.load(
        layout=qk_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    score0 = gl.dot(
        q_for_score,
        kt0_for_score,
        gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
        input_precision="tf32",
    )
    kt1_for_score = k1_smem.load(
        layout=qk_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    score1 = gl.dot(
        q_for_score,
        kt1_for_score,
        gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
        input_precision="tf32",
    )

    if MASKED:
        point_mask0 = (q_m[:, None] < SQ) & (n0[None, :] < SK)
        point_mask1 = (q_m[:, None] < SQ) & (n1[None, :] < SK)
        if CAUSAL:
            causal_limit = q_m[:, None] + SK - SQ
            point_mask0 = point_mask0 & (n0[None, :] <= causal_limit)
            point_mask1 = point_mask1 & (n1[None, :] <= causal_limit)
        score0 = gl.where(point_mask0, score0 * scale_log2, -float("inf"))
        score1 = gl.where(point_mask1, score1 * scale_log2, -float("inf"))
    else:
        score0 = score0 * scale_log2
        score1 = score1 * scale_log2

    v_fragments.index(0).store(
        gl.metax.slice(v_register, [32, BLOCK_D], [0, 0])
    )
    v_fragments.index(1).store(
        gl.metax.slice(v_register, [32, BLOCK_D], [32, 0])
    )

    has_next = n_block > 0
    next_n_block = max(n_block - 1, 0)
    next_n = next_n_block * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=gl.SliceLayout(1, global_layout),
    )
    next_valid = has_next & (next_n[:, None] < SK)
    next_k_ptrs = (
        k_head
        + next_n[:, None] * k_stride_n
        + first_d[None, :] * k_stride_d
    )
    next_k_register = gl.load(next_k_ptrs, mask=next_valid, other=0.0)

    tile_max = gl.maximum(gl.max(score0, axis=1), gl.max(score1, axis=1))
    gl.metax.barrier_shared()
    m_ij = gl.maximum(m_i, tile_max)
    if MASKED:
        row_has_key = (q_m < SQ) & (m_ij != -float("inf"))
        stable_m = gl.where(row_has_key, m_ij, 0.0)
        p0_qk = gl.where(
            point_mask0,
            gl.exp2(score0 - stable_m[:, None]),
            0.0,
        )
        p1_qk = gl.where(
            point_mask1,
            gl.exp2(score1 - stable_m[:, None]),
            0.0,
        )
        alpha = gl.where(row_has_key, gl.exp2(m_i - stable_m), 0.0)
    else:
        row_has_key = m_ij != -float("inf")
        stable_m = m_ij
        p0_qk = gl.exp2(score0 - stable_m[:, None])
        p1_qk = gl.exp2(score1 - stable_m[:, None])
        alpha = gl.exp2(m_i - stable_m)

    l_i = l_i * alpha + gl.sum(p0_qk, axis=1) + gl.sum(p1_qk, axis=1)
    alpha_for_o = gl.convert_layout(alpha[:, None], pv_mma_layout)
    acc_o = acc_o * alpha_for_o
    m_i = gl.where(row_has_key, m_ij, m_i)

    p0 = gl.convert_layout(
        p0_qk.to(v_head.dtype.element_ty),
        pv_a_layout,
        assert_trivial=True,
    )
    p1 = gl.convert_layout(
        p1_qk.to(v_head.dtype.element_ty),
        pv_a_layout,
        assert_trivial=True,
    )
    v00 = v_fragments.index(0).load(
        layout=pv_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    acc_o = gl.dot(p0, v00, acc_o, input_precision="tf32")

    v10 = v_fragments.index(1).load(
        layout=pv_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    acc_o = gl.dot(p1, v10, acc_o, input_precision="tf32")
    return next_k_register, acc_o, m_i, l_i


@gluon.jit
def flash_fwd_hdim128_generic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SQ,
    SK,
    H: gl.constexpr,
    q_stride_b,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_m,
    o_stride_d,
    lse_stride_b,
    lse_stride_h,
    lse_stride_m,
    scale_log2,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    CAUSAL: gl.constexpr,
    EVEN_MN: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_bh = gl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    o_head = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h

    qk_elements_mnk: gl.constexpr = [1, 2, 8]
    pv_elements_mnk: gl.constexpr = [1, 2, 8]

    qkv_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        qk_elements_mnk,
        0,
        [BLOCK_M, BLOCK_D],
        [1, 0],
        16,
    )
    qo_d64_page_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[2, 0, 1]
    )
    v_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        pv_elements_mnk,
        1,
        [BLOCK_N // 2, BLOCK_D],
        [1, 0],
        16,
    )
    tn_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        qk_elements_mnk,
        1,
        [BLOCK_D, BLOCK_N // 2],
        [0, 1],
        16,
    )
    qk_mma_layout: gl.constexpr = gl.MACAMmaLayout(
        version_major=2,
        version_minor=0,
        warps_per_cta=[4, 1],
        elements_mnk=qk_elements_mnk,
        col_major=1,
    )
    qk_a_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma_layout, 0)
    qk_b_layout: gl.constexpr = gl.DotOperandLayout(1, qk_mma_layout, 0)
    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_mma_layout)
    pv_mma_layout: gl.constexpr = gl.MACAMmaLayout(
        version_major=2,
        version_minor=0,
        warps_per_cta=[4, 1],
        elements_mnk=pv_elements_mnk,
        col_major=1,
    )
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma_layout, 0)
    pv_b_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma_layout, 0)
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [8, 8],
        [4, 1],
        [1, 0],
    )

    # 13b: single-bank V-delay -- 2-buffer arena for ALL paths (32KB).
    shared_arena = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_D],
        num_buffers=2,
    )
    if BLOCK_M == 64:
        qo_d64_pages = shared_arena.index(0)._reinterpret(
            q_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D // 64, 64],
            qo_d64_page_layout,
        )
        q_smem = qo_d64_pages.reshape([BLOCK_M, BLOCK_D])
        o_smem = q_smem
    else:
        q_smem = shared_arena._reinterpret(
            q_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D],
            qkv_layout,
        )
        o_smem = shared_arena._reinterpret(
            o_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D],
            qkv_layout,
        )
    k_tn_smem = shared_arena.index(0)._reinterpret(
        k_ptr.dtype.element_ty,
        [BLOCK_D, BLOCK_N],
        tn_layout,
    )
    v_fragments = shared_arena.index(1)._reinterpret(
        v_ptr.dtype.element_ty,
        [BLOCK_N // 32, 32, BLOCK_D],
        v_layout,
    )
    q_load_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, global_layout),
    )
    q_load_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    q_ptrs = (
        q_head
        + q_load_m[:, None] * q_stride_m
        + q_load_d[None, :] * q_stride_d
    )
    if EVEN_MN:
        q_register = gl.load(q_ptrs)
    else:
        q_register = gl.load(q_ptrs, mask=q_load_m[:, None] < SQ, other=0.0)

    q_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=qk_row_layout,
    )

    num_n_blocks = gl.cdiv(SK, BLOCK_N)
    if CAUSAL:
        causal_hi = (pid_m + 1) * BLOCK_M + SK - SQ
        active_n_blocks = min(num_n_blocks, max(0, gl.cdiv(causal_hi, BLOCK_N)))
    else:
        active_n_blocks = num_n_blocks
    first_n_block = active_n_blocks - 1

    first_n = first_n_block * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=gl.SliceLayout(1, global_layout),
    )
    first_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    first_k_ptrs = (
        k_head
        + first_n[:, None] * k_stride_n
        + first_d[None, :] * k_stride_d
    )
    if EVEN_MN and not CAUSAL:
        k_register = gl.load(first_k_ptrs)
    else:
        first_valid = (first_n_block >= 0) & (first_n[:, None] < SK)
        k_register = gl.load(first_k_ptrs, mask=first_valid, other=0.0)

    # C prologue: gQ -> rQ -> sQ, then publish and retain the MMA Q fragment.
    q_smem.store(q_register)
    gl.metax.barrier_shared()
    q_for_score = q_smem.load(
        layout=qk_a_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    # Every thread must finish reading sQ before the first K tile aliases it.
    gl.metax.barrier_shared()

    acc_o = gl.zeros(
        (BLOCK_M, BLOCK_D),
        gl.float32,
        layout=pv_mma_layout,
    )
    m_i = gl.full(
        (BLOCK_M,),
        -float("inf"),
        gl.float32,
        layout=qk_row_layout,
    )
    l_i = gl.zeros(
        (BLOCK_M,),
        gl.float32,
        layout=qk_row_layout,
    )

    if CAUSAL:
        if EVEN_MN:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N)
        else:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N) + 1
    elif EVEN_MN:
        masking_steps: gl.constexpr = 0
    else:
        masking_steps: gl.constexpr = 1

    masked_blocks = min(active_n_blocks, masking_steps)

    # =====================================================================
    # V3 PIPELINED PATH: BM64 + CAUSAL + EVEN_MN dense region.
    # =====================================================================
    if BLOCK_M == 64 and CAUSAL and EVEN_MN:
        n_block = max(first_n_block, 0)
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block,
            k_register,
            acc_o,
            m_i,
            l_i,
            q_for_score,
            q_m,
            k_head,
            v_head,
            k_stride_n,
            k_stride_d,
            v_stride_n,
            v_stride_d,
            SQ,
            SK,
            scale_log2,
            first_d,
            k_tn_smem,
            v_fragments,
            tn_layout,
            global_layout,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            True,
            CAUSAL,
            qk_mma_layout,
            qk_b_layout,
            pv_mma_layout,
            pv_a_layout,
            pv_b_layout,
        )

        # ---- dense-region pipeline views (2-buffer arena) ----------------
        k_ty = k_ptr.dtype.element_ty
        v_ty = v_ptr.dtype.element_ty
        k_fr = k_tn_smem._reinterpret(
            k_ty, [BLOCK_N // 32, BLOCK_D, 32], tn_layout
        )
        v_fr = v_fragments

        # Close the masked tile's smem read windows before bank-0 stores.
        gl.metax.barrier_shared()

        # S banks init to -inf; P banks init to 0 (early PV = no-op).
        s0_b = gl.full(
            (BLOCK_M, 32), -float("inf"), gl.float32, layout=qk_mma_layout
        )
        s1_b = gl.full(
            (BLOCK_M, 32), -float("inf"), gl.float32, layout=qk_mma_layout
        )
        s0_a = gl.full(
            (BLOCK_M, 32), -float("inf"), gl.float32, layout=qk_mma_layout
        )
        s1_a = gl.full(
            (BLOCK_M, 32), -float("inf"), gl.float32, layout=qk_mma_layout
        )
        # P banks: softmax output held for 1 beat before PV consumes it.
        _zero_qk = gl.zeros(
            (BLOCK_M, 32), v_ty, layout=qk_mma_layout
        )
        p0_a = gl.convert_layout(_zero_qk, pv_a_layout, assert_trivial=True)
        p1_a = gl.convert_layout(_zero_qk, pv_a_layout, assert_trivial=True)
        p0_b = gl.convert_layout(_zero_qk, pv_a_layout, assert_trivial=True)
        p1_b = gl.convert_layout(_zero_qk, pv_a_layout, assert_trivial=True)

        # Dense tiles ascend j = 0 .. dense_n-1 (n_block == j).
        dense_n = active_n_blocks - 1

        # Prologue: K[0] into registers, V[-2] = 0 (first 2 PVs are no-ops).
        dload_n = gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, global_layout)
        )
        pro_valid = (dense_n > 0) & (dload_n[:, None] < SK)
        k_reg = gl.load(
            k_head + dload_n[:, None] * k_stride_n
            + first_d[None, :] * k_stride_d,
            mask=pro_valid,
            other=0.0,
        )
        v_reg = gl.zeros((BLOCK_N, BLOCK_D), v_ty, layout=global_layout)

        n_cols = gl.arange(0, 32)
        num_pairs = gl.cdiv(dense_n + 2, 2)  # +2 drain beats
        for jj in range(0, num_pairs):
            i0 = 2 * jj
            i1 = 2 * jj + 1

            # ============ beat i0: store K[i0], V[i0-2] ============
            k_fr.index(0).store(
                gl.metax.slice(k_reg, [32, BLOCK_D], [0, 0]).permute(1, 0)
            )
            k_fr.index(1).store(
                gl.metax.slice(k_reg, [32, BLOCK_D], [32, 0]).permute(1, 0)
            )
            v_fr.index(0).store(
                gl.metax.slice(v_reg, [32, BLOCK_D], [0, 0])
            )
            v_fr.index(1).store(
                gl.metax.slice(v_reg, [32, BLOCK_D], [32, 0])
            )
            # Window A: store K[i0], V[i0-2] — pure sts
            gl.metax.iglp(config_3=2, config_5=1)
            gl.metax.barrier_shared()

            # --- Stage B: QK[i0] -> S[i0] (bank a) — lds + 2 MMA + ldg ---
            kt00 = k_fr.index(0).load(
                layout=qk_b_layout, intrinsic=True, is_constant_offs=True
            )
            s0_a = gl.dot(
                q_for_score, kt00,
                gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
                input_precision="tf32",
            )
            kt01 = k_fr.index(1).load(
                layout=qk_b_layout, intrinsic=True, is_constant_offs=True
            )
            s1_a = gl.dot(
                q_for_score, kt01,
                gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
                input_precision="tf32",
            )
            valid_w0 = i0 < dense_n
            pmask0 = (q_m[:, None] < SQ) & (n_cols[None, :] < SK) & valid_w0
            s0_a = gl.where(pmask0, s0_a * scale_log2, -float("inf"))
            s1_a = gl.where(pmask0, s1_a * scale_log2, -float("inf"))

            # --- Stage C: prefetch K[i0+1], V[i0-1] (V delayed 2 beats) ---
            pf_k_n = (i0 + 1) * BLOCK_N + dload_n
            pf_v_n = (i0 - 1) * BLOCK_N + dload_n
            pf_k_valid = (i0 + 1 < dense_n) & (pf_k_n[:, None] < SK)
            pf_v_valid = ((i0 - 1) >= 0) & ((i0 - 1) < dense_n) & (pf_v_n[:, None] < SK)
            k_reg = gl.load(
                k_head + pf_k_n[:, None] * k_stride_n
                + first_d[None, :] * k_stride_d,
                mask=pf_k_valid,
                other=0.0,
            )
            v_reg = gl.load(
                v_head + pf_v_n[:, None] * v_stride_n
                + first_d[None, :] * v_stride_d,
                mask=pf_v_valid,
                other=0.0,
            )

            # --- Stage E: PV(P[i0-2] @ V[i0-2]) — lds + 2 MMA ---
            v00 = v_fr.index(0).load(
                layout=pv_b_layout, intrinsic=True, is_constant_offs=True
            )
            acc_o = gl.dot(p0_b, v00, acc_o, input_precision="tf32")
            v10 = v_fr.index(1).load(
                layout=pv_b_layout, intrinsic=True, is_constant_offs=True
            )
            acc_o = gl.dot(p1_b, v10, acc_o, input_precision="tf32")

            # --- Stage D: softmax(S[i0-1]) -> P[i0-1] — pure Vector ---
            tile_max = gl.maximum(
                gl.max(s0_b, axis=1), gl.max(s1_b, axis=1)
            )
            m_ij = gl.maximum(m_i, tile_max)
            row_has_key = m_ij != -float("inf")
            stable_m = gl.where(row_has_key, m_ij, 0.0)
            p0_qk = gl.exp2(s0_b - stable_m[:, None])
            p1_qk = gl.exp2(s1_b - stable_m[:, None])
            alpha = gl.where(row_has_key, gl.exp2(m_i - stable_m), 0.0)
            l_i = l_i * alpha + gl.sum(p0_qk, axis=1) + gl.sum(p1_qk, axis=1)
            alpha_for_o = gl.convert_layout(alpha[:, None], pv_mma_layout)
            acc_o = acc_o * alpha_for_o
            m_i = gl.where(row_has_key, m_ij, m_i)
            p0_a = gl.convert_layout(
                p0_qk.to(v_ty), pv_a_layout, assert_trivial=True
            )
            p1_a = gl.convert_layout(
                p1_qk.to(v_ty), pv_a_layout, assert_trivial=True
            )
            # Window B: QK+PV+softmax — config_4=4 matches 4 MMAs
            gl.metax.iglp(config_0=2, config_2=2, config_4=4, config_6=2)
            gl.metax.barrier_shared()

            # ============ beat i1: store K[i1], V[i1-2]=V[i0-1] ============
            k_fr.index(0).store(
                gl.metax.slice(k_reg, [32, BLOCK_D], [0, 0]).permute(1, 0)
            )
            k_fr.index(1).store(
                gl.metax.slice(k_reg, [32, BLOCK_D], [32, 0]).permute(1, 0)
            )
            v_fr.index(0).store(
                gl.metax.slice(v_reg, [32, BLOCK_D], [0, 0])
            )
            v_fr.index(1).store(
                gl.metax.slice(v_reg, [32, BLOCK_D], [32, 0])
            )
            # Window C: store K[i1], V[i1-2] — pure sts
            gl.metax.iglp(config_3=2, config_5=1)
            gl.metax.barrier_shared()

            # --- Stage B: QK[i1] -> S[i1] (bank b) — lds + 2 MMA + ldg ---
            kt10 = k_fr.index(0).load(
                layout=qk_b_layout, intrinsic=True, is_constant_offs=True
            )
            s0_b = gl.dot(
                q_for_score, kt10,
                gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
                input_precision="tf32",
            )
            kt11 = k_fr.index(1).load(
                layout=qk_b_layout, intrinsic=True, is_constant_offs=True
            )
            s1_b = gl.dot(
                q_for_score, kt11,
                gl.zeros((BLOCK_M, 32), gl.float32, layout=qk_mma_layout),
                input_precision="tf32",
            )
            valid_w1 = i1 < dense_n
            pmask1 = (q_m[:, None] < SQ) & (n_cols[None, :] < SK) & valid_w1
            s0_b = gl.where(pmask1, s0_b * scale_log2, -float("inf"))
            s1_b = gl.where(pmask1, s1_b * scale_log2, -float("inf"))

            # --- Stage C: prefetch K[i1+1], V[i1-1]=V[i0] ---
            pf_k_n = (i1 + 1) * BLOCK_N + dload_n
            pf_v_n = (i1 - 1) * BLOCK_N + dload_n
            pf_k_valid = (i1 + 1 < dense_n) & (pf_k_n[:, None] < SK)
            pf_v_valid = ((i1 - 1) >= 0) & ((i1 - 1) < dense_n) & (pf_v_n[:, None] < SK)
            k_reg = gl.load(
                k_head + pf_k_n[:, None] * k_stride_n
                + first_d[None, :] * k_stride_d,
                mask=pf_k_valid,
                other=0.0,
            )
            v_reg = gl.load(
                v_head + pf_v_n[:, None] * v_stride_n
                + first_d[None, :] * v_stride_d,
                mask=pf_v_valid,
                other=0.0,
            )

            # --- Stage E: PV(P[i1-2]=P[i0-1] @ V[i1-2]=V[i0-1]) — lds + 2 MMA ---
            v00 = v_fr.index(0).load(
                layout=pv_b_layout, intrinsic=True, is_constant_offs=True
            )
            acc_o = gl.dot(p0_a, v00, acc_o, input_precision="tf32")
            v10 = v_fr.index(1).load(
                layout=pv_b_layout, intrinsic=True, is_constant_offs=True
            )
            acc_o = gl.dot(p1_a, v10, acc_o, input_precision="tf32")

            # --- Stage D: softmax(S[i0]) -> P[i0] — pure Vector ---
            tile_max = gl.maximum(
                gl.max(s0_a, axis=1), gl.max(s1_a, axis=1)
            )
            m_ij = gl.maximum(m_i, tile_max)
            row_has_key = m_ij != -float("inf")
            stable_m = gl.where(row_has_key, m_ij, 0.0)
            p0_qk = gl.exp2(s0_a - stable_m[:, None])
            p1_qk = gl.exp2(s1_a - stable_m[:, None])
            alpha = gl.where(row_has_key, gl.exp2(m_i - stable_m), 0.0)
            l_i = l_i * alpha + gl.sum(p0_qk, axis=1) + gl.sum(p1_qk, axis=1)
            alpha_for_o = gl.convert_layout(alpha[:, None], pv_mma_layout)
            acc_o = acc_o * alpha_for_o
            m_i = gl.where(row_has_key, m_ij, m_i)
            p0_b = gl.convert_layout(
                p0_qk.to(v_ty), pv_a_layout, assert_trivial=True
            )
            p1_b = gl.convert_layout(
                p1_qk.to(v_ty), pv_a_layout, assert_trivial=True
            )
            # Window D: QK+PV+softmax — config_4=4 matches 4 MMAs
            gl.metax.iglp(config_0=2, config_2=2, config_4=4, config_6=2)
            gl.metax.barrier_shared()

    else:
        # ================================================================
        # Original (non-pipelined) paths, unchanged from 10-*.
        # ================================================================
        if BLOCK_M == 64 and masked_blocks > 0:
            n_block = first_n_block
            k_register, acc_o, m_i, l_i = _flash_fwd_tile(
                n_block,
                k_register,
                acc_o,
                m_i,
                l_i,
                q_for_score,
                q_m,
                k_head,
                v_head,
                k_stride_n,
                k_stride_d,
                v_stride_n,
                v_stride_d,
                SQ,
                SK,
                scale_log2,
                first_d,
                k_tn_smem,
                v_fragments,
                tn_layout,
                global_layout,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                True,
                CAUSAL,
                qk_mma_layout,
                qk_b_layout,
                pv_mma_layout,
                pv_a_layout,
                pv_b_layout,
            )

        if BLOCK_M == 64:
            masked_loop_begin: gl.constexpr = 1
        else:
            masked_loop_begin: gl.constexpr = 0
        for n_iteration in range(masked_loop_begin, masked_blocks):
            n_block = first_n_block - n_iteration
            k_register, acc_o, m_i, l_i = _flash_fwd_tile(
                n_block,
                k_register,
                acc_o,
                m_i,
                l_i,
                q_for_score,
                q_m,
                k_head,
                v_head,
                k_stride_n,
                k_stride_d,
                v_stride_n,
                v_stride_d,
                SQ,
                SK,
                scale_log2,
                first_d,
                k_tn_smem,
                v_fragments,
                tn_layout,
                global_layout,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                True,
                CAUSAL,
                qk_mma_layout,
                qk_b_layout,
                pv_mma_layout,
                pv_a_layout,
                pv_b_layout,
            )

        for n_iteration in range(masked_blocks, active_n_blocks):
            n_block = first_n_block - n_iteration
            k_register, acc_o, m_i, l_i = _flash_fwd_tile(
                n_block,
                k_register,
                acc_o,
                m_i,
                l_i,
                q_for_score,
                q_m,
                k_head,
                v_head,
                k_stride_n,
                k_stride_d,
                v_stride_n,
                v_stride_d,
                SQ,
                SK,
                scale_log2,
                first_d,
                k_tn_smem,
                v_fragments,
                tn_layout,
                global_layout,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                False,
                CAUSAL,
                qk_mma_layout,
                qk_b_layout,
                pv_mma_layout,
                pv_a_layout,
                pv_b_layout,
            )

    row_has_key = (q_m < SQ) & (l_i > 0.0)
    safe_l = gl.where(row_has_key, l_i, 1.0)
    row_has_key_for_o = gl.convert_layout(
        row_has_key[:, None],
        pv_mma_layout,
    )
    safe_l_for_o = gl.convert_layout(safe_l[:, None], pv_mma_layout)
    out = gl.where(
        row_has_key_for_o,
        acc_o / safe_l_for_o,
        0.0,
    )
    lse = gl.where(
        row_has_key,
        (m_i + gl.log2(safe_l)) * 0.6931471805599453,
        float("inf"),
    )

    # C epilogue: the complete arena changes ownership from K|V to O.
    gl.metax.barrier_shared()
    o_smem.store(out.to(o_ptr.dtype.element_ty))
    gl.metax.barrier_shared()
    out_coalesced = o_smem.load(
        layout=global_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    o_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, global_layout),
    )
    o_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    o_ptrs = (
        o_head
        + o_m[:, None] * o_stride_m
        + o_d[None, :] * o_stride_d
    )
    gl.store(o_ptrs, out_coalesced, mask=o_m[:, None] < SQ)
    gl.store(
        lse_head + q_m * lse_stride_m,
        lse,
        mask=q_m < SQ,
    )


def _validate_inputs(q, k, v):
    if q.ndim != 4:
        raise ValueError("Q must be a BHSD tensor")
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim != D:
        raise ValueError(f"head dimension must be {D}, got {head_dim}")
    if k.ndim != 4 or k.shape[:2] != (batch, heads) or k.shape[3] != D:
        raise ValueError(f"invalid K shape {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if seqlen_q <= 0 or k.shape[2] <= 0:
        raise ValueError("query and key sequence lengths must be positive")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, and V must have the same dtype")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous in the head dimension")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("Q, K, and V must be on the same device")
    return batch, heads, seqlen_q, k.shape[2]


def launch_flash_fwd(q, k, v, out=None, lse=None, scale=None, causal=False):
    batch, heads, seqlen_q, seqlen_k = _validate_inputs(q, k, v)
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(
            (batch, heads, seqlen_q),
            device=q.device,
            dtype=torch.float32,
        )
    if out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("output must match Q and be contiguous in head dimension")
    if lse.shape != (batch, heads, seqlen_q) or lse.dtype != torch.float32:
        raise ValueError("LSE must be fp32 [B,H,SQ]")
    if not lse.is_contiguous():
        raise ValueError("LSE must be contiguous")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    block_m = BM64 if causal and q.dtype == torch.float16 else BM128
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    flash_fwd_hdim128_generic_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        seqlen_q,
        seqlen_k,
        heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        float(scale * LOG2E),
        BLOCK_M=block_m,
        BLOCK_N=BN,
        BLOCK_D=D,
        CAUSAL=causal,
        EVEN_MN=(seqlen_q % block_m == 0 and seqlen_k % BN == 0),
        num_warps=NUM_WARPS,
        num_stages=2,
        num_ctas=1,
        pipeline="cpasync",
    )
    return out, lse


def _prepare_flash_attn_forward(q, k, v, scale, causal):
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    def launch():
        return _flash_attn_forward(
            q_fa,
            k_fa,
            v_fa,
            0.0,
            scale,
            causal,
            (-1, -1),
        )

    (
        _,
        _,
        _,
        _,
        out_fa,
        lse_fa,
        _,
        _,
        _,
    ) = launch()
    return out_fa.transpose(1, 2), lse_fa, launch


def _run_case(seq_len, dtype, causal):
    torch.manual_seed(0)
    shape = (1, HEADS, seq_len, D)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    out, lse = launch_flash_fwd(q, k, v, scale=scale, causal=causal)
    flash_out, flash_lse, flash_launch = _prepare_flash_attn_forward(
        q, k, v, scale, causal
    )
    result = {
        "allclose": torch.allclose(
            out, flash_out, atol=3e-2, rtol=3e-2
        )
        and torch.allclose(lse, flash_lse, atol=3e-2, rtol=3e-2),
    }
    result["gluon_ms"] = triton.testing.do_bench(
        lambda: launch_flash_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            scale=scale,
            causal=causal,
        ),
        warmup=10,
        rep=30,
    )
    result["flash_ms"] = triton.testing.do_bench(
        flash_launch,
        warmup=10,
        rep=30,
    )
    result["speedup_vs_flash"] = result["flash_ms"] / result["gluon_ms"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", nargs="+", type=int, default=TEST_SEQ_LENS)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    rows = []
    results = []
    for seq_len in tuple(dict.fromkeys(args.seq_len)):
        result = _run_case(seq_len, dtype, args.causal)
        results.append(result)
        row = [
            f"1x{HEADS}x{seq_len}x{D}",
            dtype,
            args.causal,
            f"{result['gluon_ms']:.6f}",
            f"{result['flash_ms']:.6f}",
            f"{result['speedup_vs_flash']:.3f}",
            result["allclose"],
        ]
        rows.append(row)

    headers = [
        "BxHxSxD",
        "dtype",
        "causal",
        "gluon_ms",
        "_flash_attn_forward_ms",
        "gluon_speedup_vs_flash",
        "accuracy_ok",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")
    if not all(result["allclose"] for result in results):
        raise AssertionError(f"FlashAttention forward accuracy failure: {rows}")


if __name__ == "__main__":
    main()


