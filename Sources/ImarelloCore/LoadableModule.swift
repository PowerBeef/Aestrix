import Foundation

/// Heavy pipeline component that can be loaded and fully unloaded for staged memory.
public protocol LoadableModule: AnyObject {
    var moduleName: String { get }
    var isLoaded: Bool { get }

    func load() async throws
    func unload() async
}
