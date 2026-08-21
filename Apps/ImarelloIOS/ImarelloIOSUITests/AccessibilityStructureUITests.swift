import XCTest

final class AccessibilityStructureUITests: XCTestCase {
    @MainActor
    func testNativeTabsAndPrimaryControlsMeetHitTargets() throws {
        let app = launch(.empty)
        let tabBar = app.tabBars.firstMatch
        for title in ["Create", "Edit", "Gallery", "Settings"] {
            XCTAssertTrue(tabBar.buttons[title].waitForExistence(timeout: 3))
        }

        let resolution = app.buttons["create.resolution"]
        let options = app.buttons["create.options"]
        let create = app.buttons["create.action"]
        for control in [resolution, options] {
            XCTAssertTrue(control.exists)
            XCTAssertTrue(control.isHittable)
        }
        XCTAssertGreaterThanOrEqual(create.frame.width, 44)
        XCTAssertGreaterThanOrEqual(create.frame.height, 44)
    }

    @MainActor
    func testPromptRemainsReachableWithKeyboardPresented() throws {
        let app = launch(.empty)
        let prompt = app.textFields["create.prompt"]
        XCTAssertTrue(prompt.waitForExistence(timeout: 3))
        prompt.tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 2))
        XCTAssertTrue(app.buttons["create.action"].isHittable)
    }

    @MainActor
    func testResolutionControlOnlyChangesCanvasSize() throws {
        let app = launch(.empty)
        let resolution = app.buttons["create.resolution"]
        XCTAssertTrue(resolution.waitForExistence(timeout: 3))
        XCTAssertFalse(resolution.label.lowercased().contains("seed"))

        resolution.tap()
        let smallerCanvas = app.buttons["512 × 512"]
        XCTAssertTrue(smallerCanvas.waitForExistence(timeout: 2))
        smallerCanvas.tap()

        XCTAssertTrue((resolution.value as? String)?.contains("512") == true)
        XCTAssertFalse(resolution.label.lowercased().contains("seed"))
    }

    @MainActor
    func testGalleryPushesDetailAndEditRoutesToEditWorkspace() throws {
        let app = launch(.library)
        app.tabBars.buttons["Gallery"].tap()
        let cell = app.buttons["gallery.cell.woman_i2i"]
        XCTAssertTrue(cell.waitForExistence(timeout: 3))
        cell.tap()
        XCTAssertTrue(app.navigationBars.firstMatch.waitForExistence(timeout: 3))
        XCTAssertFalse(app.tabBars.firstMatch.exists)

        let edit = app.buttons["print.edit"]
        XCTAssertTrue(edit.exists)
        edit.tap()
        XCTAssertTrue(app.buttons["edit.action"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["edit.change-source"].exists)
    }

    @MainActor
    func testEditStartsWithNewestFirstSourcePicker() throws {
        let app = launch(.library)
        app.tabBars.buttons["Edit"].tap()
        let featured = app.buttons["edit.featured-source"]
        XCTAssertTrue(featured.waitForExistence(timeout: 3))
        XCTAssertTrue(featured.label.contains("seed 43"))
        XCTAssertFalse(app.buttons["edit.source.woman_i2i"].exists)
        XCTAssertTrue(app.buttons["edit.source.woman_t2i"].exists)
        featured.tap()
        XCTAssertTrue(app.buttons["edit.action"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["edit.options"].exists)
        XCTAssertTrue((app.buttons["edit.change-source"].value as? String)?.contains("seed 43") == true)
    }

    @MainActor
    func testEditRecentSourceEntersTheCorrectWorkspace() throws {
        let app = launch(.library)
        app.tabBars.buttons["Edit"].tap()
        let source = app.buttons["edit.source.woman_t2i"]
        XCTAssertTrue(source.waitForExistence(timeout: 3))
        source.tap()
        let changeSource = app.buttons["edit.change-source"]
        XCTAssertTrue(changeSource.waitForExistence(timeout: 3))
        XCTAssertTrue((changeSource.value as? String)?.contains("seed 42") == true)
    }

    @MainActor
    func testEditSourcePickerHandlesEmptySingleMissingAndManyStates() throws {
        let empty = launch(.empty)
        empty.tabBars.buttons["Edit"].tap()
        XCTAssertTrue(empty.descendants(matching: .any)["edit.empty"].waitForExistence(timeout: 3))
        empty.terminate()

        let single = launch(.singleSource)
        single.tabBars.buttons["Edit"].tap()
        XCTAssertTrue(single.buttons["edit.featured-source"].waitForExistence(timeout: 3))
        XCTAssertFalse(single.descendants(matching: .any)["edit.source-strip"].exists)
        XCTAssertTrue(single.staticTexts["Tap the image to begin editing."].exists)
        single.terminate()

        let missing = launch(.sourceLoadFailure)
        missing.tabBars.buttons["Edit"].tap()
        XCTAssertTrue(
            missing.descendants(matching: .any)["edit.source-unavailable"]
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(missing.staticTexts["No other images are available."].exists)
        missing.terminate()

        let many = launch(.manySources)
        many.tabBars.buttons["Edit"].tap()
        XCTAssertTrue(many.buttons["edit.featured-source"].waitForExistence(timeout: 3))
        XCTAssertTrue(many.buttons["edit.source.portrait_5"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testGalleryUsesCreateMatchedHeaderAndKeepsDenseGrid() throws {
        let app = launch(.library)
        app.tabBars.buttons["Gallery"].tap()
        XCTAssertTrue(app.navigationBars["Gallery"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["2 images"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["gallery.grid"].exists)
        XCTAssertTrue(app.buttons["gallery.cell.woman_i2i"].exists)
        XCTAssertTrue(app.buttons["gallery.cell.woman_t2i"].exists)
    }

    @MainActor
    func testSettingsShowsUsefulEssentials() throws {
        let app = launch(.library)
        app.tabBars.buttons["Settings"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["Device and Model"].exists)
        XCTAssertTrue(app.staticTexts["Gallery"].exists)
        XCTAssertTrue(app.staticTexts["Photos"].exists)
    }

    @MainActor
    func testRunningAndFailureAccessoriesExposeOwnedActions() throws {
        let running = launch(.running)
        XCTAssertTrue(running.buttons["generation.cancel"].waitForExistence(timeout: 3))
        XCTAssertTrue(running.staticTexts["Creating 2/4"].exists)
        running.terminate()

        let failed = launch(.failed)
        XCTAssertTrue(failed.buttons["generation.retry"].waitForExistence(timeout: 3))
        XCTAssertTrue(failed.buttons["Dismiss error"].exists)
    }

    @MainActor
    private func launch(_ scenario: Scenario) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["--ui-test-scenario", scenario.rawValue]
        app.launch()
        return app
    }
}

private enum Scenario: String {
    case empty
    case library
    case singleSource = "single-source"
    case sourceLoadFailure = "source-load-failure"
    case manySources = "many-sources"
    case pendingEdit = "pending-edit"
    case running
    case failed
}

private extension XCUIElement {
    @MainActor
    func clearAndType(_ text: String) {
        guard let current = value as? String else {
            typeText(text)
            return
        }
        typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count))
        typeText(text)
    }
}
