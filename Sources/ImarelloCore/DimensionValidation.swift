public enum DimensionValidation {
    public static func validate(width: Int, height: Int, maxSide: Int, tier: DeviceTier) throws {
        guard width > 0, height > 0 else {
            throw ImarelloError.invalidDimensions(width: width, height: height, reason: "must be positive")
        }
        guard width % 16 == 0, height % 16 == 0 else {
            throw ImarelloError.invalidDimensions(
                width: width,
                height: height,
                reason: "width and height must be multiples of 16 (latent /16 packing)"
            )
        }
        let side = max(width, height)
        if side > maxSide {
            throw ImarelloError.resolutionExceedsTier(side: side, tier: tier, maxSide: maxSide)
        }
    }
}

public enum RequestValidation {
    public static func validate(steps: Int) throws {
        guard steps >= 1 else { throw ImarelloError.invalidSteps(steps) }
    }

    public static func validate(_ request: T2IRequest) throws {
        try validate(steps: request.steps)
        guard request.guidance.isFinite else {
            throw ImarelloError.invalidRequest("guidance must be finite")
        }
        guard request.guidance == 1 else {
            throw ImarelloError.invalidRequest("Klein guidance is locked to 1.0")
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImarelloError.invalidRequest("prompt must not be empty")
        }
        if let padKeep = request.padKeep, !(0 ... 512).contains(padKeep) {
            throw ImarelloError.invalidRequest("padKeep must be between 0 and 512")
        }
    }

    public static func validate(_ request: I2IRequest) throws {
        try validate(steps: request.steps)
        guard request.strength.isFinite, request.strength > 0, request.strength <= 1 else {
            throw ImarelloError.invalidStrength(request.strength)
        }
        guard request.guidance.isFinite else {
            throw ImarelloError.invalidRequest("guidance must be finite")
        }
        guard request.guidance == 1 else {
            throw ImarelloError.invalidRequest("Klein guidance is locked to 1.0")
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImarelloError.invalidRequest("prompt must not be empty")
        }
    }
}
