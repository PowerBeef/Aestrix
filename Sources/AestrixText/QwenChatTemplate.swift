import Foundation

/// Fixed Qwen3 chat template used by FLUX.2-klein prompt encoding.
///
/// Matches diffusers `Flux2KleinPipeline._get_qwen3_prompt_embeds`:
/// ```
/// tokenizer.apply_chat_template(
///   [{"role":"user","content": prompt}],
///   tokenize=False,
///   add_generation_prompt=True,
///   enable_thinking=False,
/// )
/// ```
/// which expands to an empty think block before the (unfilled) assistant turn.
public enum QwenChatTemplate {
    public static let imStart = "<|im_start|>"
    public static let imEnd = "<|im_end|>"
    public static let thinkOpen = "<think>"
    public static let thinkClose = "</think>"

    /// Format a single user prompt for the text encoder.
    public static func format(
        userPrompt: String,
        enableThinking: Bool = false,
        addGenerationPrompt: Bool = true
    ) -> String {
        var text = "\(imStart)user\n\(userPrompt)\(imEnd)\n"
        if addGenerationPrompt {
            text += "\(imStart)assistant\n"
            if !enableThinking {
                // Exact empty-think block required for klein quality (BFL / diffusers).
                text += "\(thinkOpen)\n\n\(thinkClose)\n\n"
            }
        }
        return text
    }
}
