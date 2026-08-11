public enum DimensionValidation {
    public static func validate(width: Int, height: Int, maxSide: Int, tier: DeviceTier) throws {
        guard width > 0, height > 0 else {
            throw AestrixError.invalidDimensions(width: width, height: height, reason: "must be positive")
        }
        guard width % 16 == 0, height % 16 == 0 else {
            throw AestrixError.invalidDimensions(
                width: width,
                height: height,
                reason: "width and height must be multiples of 16 (latent /16 packing)"
            )
        }
        let side = max(width, height)
        if side > maxSide {
            throw AestrixError.resolutionExceedsTier(side: side, tier: tier, maxSide: maxSide)
        }
    }
}
