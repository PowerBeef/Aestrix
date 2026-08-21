import Foundation

/// Versioned contract between Swift dispatch code and the build-time Direct
/// metallib. Private MLX symbols remain a separate, explicitly verified ABI.
public enum DirectKernelABI {
    public static let version = 1
    public static let metal4ArgumentTableBufferLimit = 31

    public static let requiredDirectSymbols = [
        "dd_ln_mod_prescale",
        "dd_rmsnorm_pitched",
        "dd_rope_interleaved",
        "dd_scale_cast_pitched",
        "dd_swiglu_pitched",
        "dd_scale_inplace",
        "dd_silu_f32",
        "dd_cast_prescale",
        "dd_cast_postscale",
        "dd_gate_add",
        "dq_rmsnorm_half",
        "dq_rope_half",
        "dq_silu_mul_half",
        "dq_add_half",
        "dq_rmsnorm_bfloat",
        "dq_rope_bfloat",
        "dq_silu_mul_bfloat",
        "dq_add_bfloat",
        "dv_gn_partial",
        "dv_gn_finalize",
        "dv_gn_apply",
        "dv_bias_act",
        "dv_upsample2",
        "dv_add",
        "dv_matmul_nt",
        "dv_matmul_nn",
        "dv_softmax_rows",
    ].sorted()

    public static let requiredMLXSymbols = [
        "affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0",
        "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
        "affine_qmm_t_float_gs_64_b_4_alN_true_batch_0",
        "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
        "steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1_maskbfloat16",
    ].sorted()

    /// Already packaged by the pinned sibling fork. These are qualified per
    /// device/profile and are not compatibility-backend prerequisites.
    public static let naxAttentionSymbols = [
        "steel_attention_float16_bq64_bk32_bd128_wm4_wn1_maskfloat16",
        "steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1_maskbfloat16",
    ].sorted()

    public static let requiredFunctionConstants = [
        "attention.alignedQueries": AttentionFunctionConstant.alignedQueries,
        "attention.alignedKeys": AttentionFunctionConstant.alignedKeys,
        "attention.hasMask": AttentionFunctionConstant.hasMask,
        "attention.causal": AttentionFunctionConstant.causal,
        "attention.doReads": AttentionFunctionConstant.doReads,
    ]

    /// MLX Steel attention function constants used by Direct.
    public enum AttentionFunctionConstant {
        public static let alignedQueries = 200
        public static let alignedKeys = 201
        public static let hasMask = 300
        public static let causal = 301
        public static let doReads = 302
    }
}
