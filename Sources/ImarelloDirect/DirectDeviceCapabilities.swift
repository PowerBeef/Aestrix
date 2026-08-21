import Foundation
import Metal

public struct DirectKernelInventory: Sendable, Equatable {
    public var requiredCompatibilitySymbolsPresent: Bool
    public var missingCompatibilitySymbols: [String]
    public var naxAttentionSymbolsPresent: [String]
    public var naxAttentionSymbolsMissing: [String]
}

public enum DirectDeviceCapabilities {
    public static func inventory(mlxMetallibData: Data) -> DirectKernelInventory {
        func present(_ symbol: String) -> Bool {
            mlxMetallibData.range(of: Data(symbol.utf8)) != nil
        }
        let missingCompatibility = DirectKernelABI.requiredMLXSymbols.filter { !present($0) }
        let naxPresent = DirectKernelABI.naxAttentionSymbols.filter(present)
        return DirectKernelInventory(
            requiredCompatibilitySymbolsPresent: missingCompatibility.isEmpty,
            missingCompatibilitySymbols: missingCompatibility,
            naxAttentionSymbolsPresent: naxPresent,
            naxAttentionSymbolsMissing: DirectKernelABI.naxAttentionSymbols.filter { !present($0) })
    }

    /// Native sub-byte tensor types are SDK/runtime gated independently from
    /// hardware/profile qualification. Compatibility execution never depends
    /// on this result.
    public static var supportsNativeInt4TensorType: Bool {
        if #available(macOS 26.4, iOS 26.4, *) {
            #if targetEnvironment(simulator)
            return false
            #else
            return true
            #endif
        }
        return false
    }

    #if !targetEnvironment(simulator)
    public static var nativeInt4TensorType: MTLTensorDataType? {
        guard supportsNativeInt4TensorType else { return nil }
        if #available(macOS 26.4, iOS 26.4, *) {
            return .int4
        }
        return nil
    }
    #endif
}
