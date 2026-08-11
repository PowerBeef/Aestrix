import Testing
import AestrixText
import AestrixCore

@Suite("Qwen chat template")
struct QwenChatTemplateTests {
    @Test func emptyThinkBlockForKlein() {
        let text = QwenChatTemplate.format(userPrompt: "a red cube", enableThinking: false)
        #expect(text.contains("<|im_start|>user\n"))
        #expect(text.contains("a red cube"))
        #expect(text.contains("<|im_end|>\n"))
        #expect(text.contains("<|im_start|>assistant\n"))
        #expect(text.contains("<think>\n\n</think>\n\n"))
    }

    @Test func thinkingEnabledOmitsEmptyBlock() {
        let text = QwenChatTemplate.format(userPrompt: "hi", enableThinking: true)
        #expect(!text.contains("<think>"))
        #expect(text.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test func configTaps() {
        let c = Qwen3Config.klein4B
        #expect(c.tapHiddenStateIndices == [9, 18, 27])
        #expect(c.layersNeededForTaps == 27)
        #expect(c.jointAttentionDim == ModelConstants.jointAttentionDim)
    }
}
