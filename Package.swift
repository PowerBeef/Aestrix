// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Imarello",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
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
        .executable(name: "imarello", targets: ["ImarelloCLI"]),
    ],
    dependencies: [
        // Pinned exact deliberately: kernel/runtime ABI must match the metallib
        // Scripts/ensure-metallib.sh builds. Bump only with validation
        // (CLAUDE.md), then rebuild the metallib (the script detects the bump).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
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
