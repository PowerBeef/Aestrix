/// How model weights stay resident across pipeline stages.
public enum MemoryPolicy: String, Sendable, Codable, CaseIterable {
    /// At most one heavy module (TE / DiT / VAE) resident. Default for all devices.
    case staged
    /// Staged + lower res pressure + optional DiT block streaming.
    case stagedAggressive
    /// Keep modules warm after first use (high-RAM interactive only).
    case resident
}
