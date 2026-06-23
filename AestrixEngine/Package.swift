// swift-tools-version: 6.0
// AestrixEngine — optimized MLX inference engine for FLUX.2-klein-4B on iOS.
import PackageDescription

let package = Package(
    name: "AestrixEngine",
    platforms: [
        // Library floor matches MLX-Swift (iOS 17 / macOS 14). iOS-26-only APIs used
        // inside the library are gated with `@available`. The authoritative iOS 26
        // device floor is the host app's deployment target (set in AestrixDemo).
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "AestrixEngine", targets: ["AestrixEngine"]),
    ],
    dependencies: [
        // Pinned MLX-Swift (see plan risk: "research not production" → pin a version).
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
    ],
    targets: [
        .target(
            name: "AestrixEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "AestrixEngineTests",
            dependencies: ["AestrixEngine"]
        ),
    ]
)
