import Foundation

public enum DirectAttentionBackend: String, Codable, Sendable, CaseIterable {
    case steel
    case nax
}

public struct DirectAttentionProfile: Sendable, Equatable {
    public var backend: DirectAttentionBackend
    public var functionName: String
    public var blockQueries: Int
    public var blockKeys: Int
    public var headDimension: Int
    public var warpM: Int
    public var warpN: Int

    public static let steelF16 = DirectAttentionProfile(
        backend: .steel,
        functionName: "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
        blockQueries: 32,
        blockKeys: 16,
        headDimension: 128,
        warpM: 4,
        warpN: 1)

    public static let naxF16 = DirectAttentionProfile(
        backend: .nax,
        functionName: "steel_attention_float16_bq64_bk32_bd128_wm4_wn1_maskfloat16",
        blockQueries: 64,
        blockKeys: 32,
        headDimension: 128,
        warpM: 4,
        warpN: 1)

    public static let qualificationJointLengths = [1_536, 2_816, 4_608]
}
