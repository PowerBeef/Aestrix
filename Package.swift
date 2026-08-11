// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Aestrix",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "AestrixCore", targets: ["AestrixCore"]),
        .library(name: "AestrixWeights", targets: ["AestrixWeights"]),
        .library(name: "AestrixText", targets: ["AestrixText"]),
        .library(name: "AestrixDiT", targets: ["AestrixDiT"]),
        .library(name: "AestrixVAE", targets: ["AestrixVAE"]),
        .library(name: "AestrixRuntime", targets: ["AestrixRuntime"]),
        .library(name: "AestrixEval", targets: ["AestrixEval"]),
        .library(name: "AestrixBench", targets: ["AestrixBench"]),
        .executable(name: "aestrix", targets: ["AestrixCLI"]),
    ],
    dependencies: [
        // mlx-swift is declared for Phases 1+ (model port). Phase 0 stubs do not
        // link it so `aestrix mem-selftest` works without metallib. Uncomment
        // product deps on Text/DiT/VAE when implementing those modules.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "AestrixCore",
            dependencies: []
        ),
        .target(
            name: "AestrixWeights",
            dependencies: [
                "AestrixCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "AestrixText",
            dependencies: [
                "AestrixCore",
                "AestrixWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "AestrixDiT",
            dependencies: [
                "AestrixCore",
                "AestrixWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "AestrixVAE",
            dependencies: [
                "AestrixCore",
                "AestrixWeights",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "AestrixRuntime",
            dependencies: [
                "AestrixCore",
                "AestrixWeights",
                "AestrixText",
                "AestrixDiT",
                "AestrixVAE",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            linkerSettings: [
                .linkedFramework("Vision"),
            ]
        ),
        .target(
            name: "AestrixEval",
            dependencies: [
                "AestrixCore",
            ],
            linkerSettings: [
                .linkedFramework("Vision"),
                .linkedFramework("CoreML"),
            ]
        ),
        .target(
            name: "AestrixBench",
            dependencies: [
                "AestrixCore",
                "AestrixRuntime",
                "AestrixDiT",
                "AestrixEval",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "AestrixCLI",
            dependencies: [
                "AestrixCore",
                "AestrixRuntime",
                "AestrixDiT",
                "AestrixEval",
                "AestrixBench",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            // Placeholder default Metal library for MLX when using `swift build`
            // (xcodebuild may also emit a fuller metallib; colocated mlx.metallib is enough for JIT).
            resources: [
                .copy("Resources/mlx.metallib"),
            ]
        ),
        .testTarget(
            name: "AestrixCoreTests",
            dependencies: ["AestrixCore", "AestrixRuntime"]
        ),
        .testTarget(
            name: "AestrixTextTests",
            dependencies: [
                "AestrixText",
                "AestrixCore",
            ]
        ),
        .testTarget(
            name: "AestrixDiTTests",
            dependencies: [
                "AestrixDiT",
                "AestrixCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "AestrixEvalTests",
            dependencies: [
                "AestrixEval",
                "AestrixCore",
            ]
        ),
        .testTarget(
            name: "AestrixBenchTests",
            dependencies: [
                "AestrixBench",
                "AestrixCore",
            ]
        ),
        .testTarget(
            name: "AestrixVAETests",
            dependencies: [
                "AestrixVAE",
                "AestrixCore",
            ]
        ),
    ]
)
