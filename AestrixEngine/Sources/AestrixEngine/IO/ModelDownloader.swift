import Foundation

/// One file in a model repo: its path relative to the repo root and its exact byte size.
public struct ModelFile: Sendable, Hashable {
    public let relativePath: String
    public let size: Int64
    public init(relativePath: String, size: Int64) {
        self.relativePath = relativePath
        self.size = size
    }
}

/// Progress for a single in-flight download.
public struct DownloadProgress: Sendable {
    public let file: String
    public let fileIndex: Int      // 0-based
    public let fileCount: Int
    public let bytesWritten: Int64
    public let totalBytes: Int64
    /// Per-file fraction complete (0…1).
    public var fraction: Double { totalBytes > 0 ? Double(bytesWritten) / Double(totalBytes) : 0 }
    /// Overall fraction across all files (0…1).
    public var overallFraction: Double {
        guard fileCount > 0, totalBytes > 0 else { return 0 }
        let perFile = 1.0 / Double(fileCount)
        return (Double(fileIndex) + fraction) * perFile
    }
}

/// Manifest for [`mlx-community/flux2-klein-4b-4bit`](https://hf.co/mlx-community/flux2-klein-4b-4bit).
///
/// Sizes are pinned from the HF API so we can size-verify downloads (resumable 4.6 GB over
/// flaky mobile needs integrity checks). Quantization: 4-bit, group_size 64 (mflux 0.17.5).
public enum Flux2Klein4B4Bit {
    public static let repoId = "mlx-community/flux2-klein-4b-4bit"
    public static let baseURL = URL(string: "https://huggingface.co/mlx-community/flux2-klein-4b-4bit/resolve/main/")!

    public static let files: [ModelFile] = [
        .init(relativePath: "text_encoder/0.safetensors", size: 2_135_435_122),
        .init(relativePath: "text_encoder/1.safetensors", size: 127_582_140),
        .init(relativePath: "text_encoder/model.safetensors.index.json", size: 51_369),
        .init(relativePath: "tokenizer/chat_template.jinja", size: 4_168),
        .init(relativePath: "tokenizer/tokenizer.json", size: 11_422_650),
        .init(relativePath: "tokenizer/tokenizer_config.json", size: 703),
        .init(relativePath: "transformer/0.safetensors", size: 2_145_323_732),
        .init(relativePath: "transformer/1.safetensors", size: 34_727_761),
        .init(relativePath: "transformer/model.safetensors.index.json", size: 26_945),
        .init(relativePath: "vae/0.safetensors", size: 165_107_943),
        .init(relativePath: "vae/model.safetensors.index.json", size: 17_584),
    ]

    /// Total bytes across all files (~4.6 GB).
    public static var totalSize: Int64 { files.reduce(0) { $0 + $1.size } }

    /// The subset needed to exercise the tokenizer (M1 test fixtures).
    public static let tokenizerFiles: [ModelFile] = files.filter { $0.relativePath.hasPrefix("tokenizer/") }
    /// The subset needed to exercise the VAE weight loader (M1 test fixtures).
    public static let vaeFiles: [ModelFile] = files.filter { $0.relativePath.hasPrefix("vae/") }
    /// The subset needed to exercise the Qwen3 text encoder (M3).
    public static let textEncoderFiles: [ModelFile] = files.filter { $0.relativePath.hasPrefix("text_encoder/") }
}

/// Downloads model files from Hugging Face with progress, size verification, skip-if-complete,
/// and a single resume-data retry on failure.
///
/// Single-flight: downloads run sequentially (one `URLSessionDownloadTask` at a time), which is
/// fine for staged loading and avoids saturating mobile bandwidth. Background-session support
/// (downloads continuing while the app is backgrounded) is a planned follow-up.
public actor ModelDownloader {
    public let files: [ModelFile]
    public let baseURL: URL
    public let destinationRoot: URL

    private let session: URLSession
    private let delegate = DownloadDelegate()

    public init(
        files: [ModelFile] = Flux2Klein4B4Bit.files,
        baseURL: URL = Flux2Klein4B4Bit.baseURL,
        destinationRoot: URL
    ) {
        self.files = files
        self.baseURL = baseURL
        self.destinationRoot = destinationRoot
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 6 * 3600  // large files on slow links
        // delegate is retained by the session for its lifetime.
        self.session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
    }

    // MARK: - public API

    /// `true` if the file is present at its destination and its size matches the manifest.
    public func isComplete(_ file: ModelFile) -> Bool {
        let url = destination(for: file)
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int64 else { return false }
        return size == file.size
    }

    /// Files not yet fully downloaded.
    public func missingFiles() -> [ModelFile] {
        files.filter { !isComplete($0) }
    }

    /// Download every missing file, reporting per-file progress. Files already complete are skipped.
    public func ensureDownloaded(onProgress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }) async throws {
        let count = files.count
        for (index, file) in files.enumerated() where !isComplete(file) {
            try await download(file, fileIndex: index, fileCount: count, onProgress: onProgress)
        }
    }

    // MARK: - internals

    private func destination(for file: ModelFile) -> URL {
        destinationRoot.appendingPathComponent(file.relativePath)
    }

    private func download(
        _ file: ModelFile,
        fileIndex: Int,
        fileCount: Int,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        let dest = destination(for: file)
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

        let source = baseURL.appendingPathComponent(file.relativePath)

        do {
            let tmp = try await runTask(source: source, resumeData: nil, file: file,
                                        fileIndex: fileIndex, fileCount: fileCount, onProgress: onProgress)
            try verifyAndMove(tmp: tmp, to: dest, expected: file.size)
        } catch {
            // One resume-data retry on failure (flaky mobile links).
            if let resume = delegate.consumeResumeData() {
                AestrixLog.notice("Retrying \(file.relativePath) with resume data…")
                let tmp = try await runTask(source: source, resumeData: resume, file: file,
                                            fileIndex: fileIndex, fileCount: fileCount, onProgress: onProgress)
                try verifyAndMove(tmp: tmp, to: dest, expected: file.size)
            } else {
                throw AestrixError.downloadFailed("\(file.relativePath): \(error.localizedDescription)")
            }
        }
    }

    private func runTask(
        source: URL,
        resumeData: Data?,
        file: ModelFile,
        fileIndex: Int,
        fileCount: Int,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws -> URL {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<URL, Error>) in
            delegate.prepare(continuation: cont) { written, total in
                onProgress(DownloadProgress(
                    file: file.relativePath, fileIndex: fileIndex, fileCount: fileCount,
                    bytesWritten: written, totalBytes: total > 0 ? total : file.size))
            }
            let task: URLSessionDownloadTask
            if let resumeData {
                task = session.downloadTask(withResumeData: resumeData)
            } else {
                task = session.downloadTask(with: URLRequest(url: source))
            }
            task.taskDescription = file.relativePath
            task.resume()
        }
    }

    private func verifyAndMove(tmp: URL, to dest: URL, expected: Int64) throws {
        let actual = (try? FileManager.default.attributesOfItem(atPath: tmp.path)[.size] as? Int64) ?? -1
        guard actual == expected else {
            try? FileManager.default.removeItem(at: tmp)
            throw AestrixError.downloadFailed("size mismatch: got \(actual), expected \(expected)")
        }
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.moveItem(at: tmp, to: dest)
    }
}

/// Single-flight delegate for `ModelDownloader`. Bridges `URLSessionDownloadDelegate` callbacks
/// to an async continuation and a progress handler; captures resume data on failure for retry.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<URL, Error>?
    private var progress: ((Int64, Int64) -> Void)?
    private var downloadedURL: URL?
    private var resumeData: Data?
    private let lock = NSLock()

    func prepare(continuation: CheckedContinuation<URL, Error>, progress: @escaping (Int64, Int64) -> Void) {
        lock.lock()
        self.continuation = continuation
        self.progress = progress
        self.downloadedURL = nil
        self.resumeData = nil
        lock.unlock()
    }

    func consumeResumeData() -> Data? {
        lock.lock(); defer { lock.unlock() }
        let data = resumeData
        resumeData = nil
        return data
    }

    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData _: Int64, totalBytesWritten written: Int64,
                    totalBytesExpectedToWrite expected: Int64) {
        progress?(written, expected)
    }

    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The system deletes `location` after this returns — move it somewhere we own.
        let moved = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: moved)
        } catch {
            try? FileManager.default.removeItem(at: location)
        }
        lock.lock(); self.downloadedURL = moved; lock.unlock()
    }

    func urlSession(_: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let cont = continuation
        let url = downloadedURL
        if error != nil {
            // Universal (works on all OS versions): the resume-data blob lives in userInfo.
            resumeData = (error as? NSError)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        }
        continuation = nil
        progress = nil
        lock.unlock()

        if let error {
            cont?.resume(throwing: error)
        } else if let url {
            cont?.resume(returning: url)
        } else {
            cont?.resume(throwing: AestrixError.downloadFailed("download produced no file"))
        }
    }
}
