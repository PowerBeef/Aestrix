// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Imarello",
    platforms: [
        // 26.2 floor (2026-08-18): Metal 4 as a single API target and the
        // MLX NAX path everywhere. Every Apple Silicon Mac runs macOS 26,
        // so no supported chip is dropped; NAX stays runtime-gated to
        // A19/M5-class hardware.
        .macOS("26.2"),
        .iOS("26.2"),
    ],
    products: [
        .library(name: "ImarelloCore", targets: ["ImarelloCore"]),
        .library(name: "ImarelloWeights", targets: ["ImarelloWeights"]),
        .library(name: "ImarelloText", targets: ["ImarelloText"]),
        .library(name: "ImarelloDiT", targets: ["ImarelloDiT"]),
        .library(name: "ImarelloVAE", targets: ["ImarelloVAE"]),
        .library(name: "ImarelloRuntime", targets: ["ImarelloRuntime"]),
        .library(name: "ImarelloEval", targets: ["ImarelloEval"]),
        .library(name: "ImarelloBench", targets: ["ImarelloBench"]),
        .library(name: "ImarelloDirect", targets: ["ImarelloDirect"]),
        .executable(name: "imarello", targets: ["ImarelloCLI"]),
    ],
    dependencies: [
        // Pinned exact deliberately: kernel/runtime ABI must match the metallib
        // Scripts/ensure-metallib.sh builds. Bump only with validation
        // (CLAUDE.md), then rebuild the metallib (the script detects the bump).
        // Imarello fork of mlx-swift: core v0.32.1 base build (see the fork's
        // imarello/core-0.32.1 branch; upstream is stuck pre-0.32 — issue #446).
        .package(url: "https://github.com/PowerBeef/mlx-swift.git", revision: "079609aa3da8550ff5b98b9fefeb14288043541c"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "ImarelloCore",
            dependencies: []
        ),
        .target(
            name: "ImarelloWeights",
            dependencies: [
                "ImarelloCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ImarelloDirect",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                "ImarelloText",
                "ImarelloDiT",
                "ImarelloRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ImarelloText",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ImarelloDiT",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ImarelloVAE",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ImarelloRuntime",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                "ImarelloText",
                "ImarelloDiT",
                "ImarelloVAE",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            linkerSettings: [
                .linkedFramework("Vision"),
            ]
        ),
        .target(
            name: "ImarelloEval",
            dependencies: [
                "ImarelloCore",
            ],
            linkerSettings: [
                .linkedFramework("Vision"),
                .linkedFramework("CoreML"),
            ]
        ),
        .target(
            name: "ImarelloBench",
            dependencies: [
                "ImarelloCore",
                "ImarelloRuntime",
                "ImarelloDiT",
                "ImarelloVAE",
                "ImarelloEval",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "ImarelloCLI",
            dependencies: [
                "ImarelloCore",
                "ImarelloWeights",
                "ImarelloRuntime",
                "ImarelloDiT",
                "ImarelloVAE",
                "ImarelloEval",
                "ImarelloBench",
                "ImarelloDirect",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            // Placeholder default Metal library for MLX when using `swift build`
            // (xcodebuild may also emit a fuller metallib; colocated mlx.metallib is enough for JIT).
            resources: [
                .copy("Resources/mlx.metallib"),
            ]
        ),
        .testTarget(
            name: "ImarelloCoreTests",
            dependencies: ["ImarelloCore", "ImarelloWeights", "ImarelloRuntime"]
        ),
        .testTarget(
            name: "ImarelloTextTests",
            dependencies: [
                "ImarelloText",
                "ImarelloCore",
            ],
            resources: [
                .copy("Fixtures")
            ]
        ),
        .testTarget(
            name: "ImarelloDiTTests",
            dependencies: [
                "ImarelloDiT",
                "ImarelloCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "ImarelloEvalTests",
            dependencies: [
                "ImarelloEval",
                "ImarelloCore",
            ]
        ),
        .testTarget(
            name: "ImarelloBenchTests",
            dependencies: [
                "ImarelloBench",
                "ImarelloCore",
                "ImarelloDiT",
            ]
        ),
        .testTarget(
            name: "ImarelloVAETests",
            dependencies: [
                "ImarelloVAE",
                "ImarelloCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
