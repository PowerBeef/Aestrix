import Foundation
import ImarelloCore

/// Byte-level BPE tokenizer for Qwen2/Qwen3 `tokenizer.json` (no external deps).
public final class QwenTokenizer: @unchecked Sendable {
    public let vocab: [String: Int]
    public let idToToken: [Int: String]
    public let merges: [(String, String)]
    public let bpeRanks: [String: Int]
    public let specialTokens: [String: Int]
    public let specialTokenSet: Set<String>
    public let padTokenId: Int
    public let eosTokenId: Int
    public let byteEncoder: [UInt8: Character]
    public let byteDecoder: [Character: UInt8]

    public init(
        vocab: [String: Int],
        merges: [(String, String)],
        specialTokens: [String: Int],
        padTokenId: Int,
        eosTokenId: Int
    ) {
        self.vocab = vocab
        self.merges = merges
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (i, pair) in merges.enumerated() {
            ranks["\(pair.0) \(pair.1)"] = i
        }
        self.bpeRanks = ranks
        self.specialTokens = specialTokens
        self.specialTokenSet = Set(specialTokens.keys)
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId

        var inv: [Int: String] = [:]
        inv.reserveCapacity(vocab.count)
        for (tok, id) in vocab {
            inv[id] = tok
        }
        for (tok, id) in specialTokens {
            inv[id] = tok
        }
        self.idToToken = inv

        let (enc, dec) = Self.bytesToUnicode()
        self.byteEncoder = enc
        self.byteDecoder = dec
    }

    /// Load from a snapshot `tokenizer/` directory (`tokenizer.json` + optional config).
    public static func load(from directory: URL) throws -> QwenTokenizer {
        let jsonURL = directory.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: jsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ImarelloError.unsupportedWeightFormat("tokenizer.json root")
        }

        guard
            let model = root["model"] as? [String: Any],
            let vocabObj = model["vocab"] as? [String: Any]
        else {
            throw ImarelloError.unsupportedWeightFormat("tokenizer.json missing model.vocab")
        }

        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(vocabObj.count)
        for (k, v) in vocabObj {
            if let i = v as? Int {
                vocab[k] = i
            } else if let n = v as? NSNumber {
                vocab[k] = n.intValue
            }
        }

        var merges: [(String, String)] = []
        if let mergeList = model["merges"] as? [Any] {
            merges.reserveCapacity(mergeList.count)
            for item in mergeList {
                if let s = item as? String {
                    let parts = s.split(
                        separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
                    if parts.count == 2 {
                        merges.append((String(parts[0]), String(parts[1])))
                    }
                } else if let arr = item as? [String], arr.count == 2 {
                    merges.append((arr[0], arr[1]))
                } else if let arr = item as? [Any], arr.count == 2,
                    let a = arr[0] as? String, let b = arr[1] as? String
                {
                    merges.append((a, b))
                }
            }
        }

        var specials: [String: Int] = [:]
        if let added = root["added_tokens"] as? [[String: Any]] {
            for tok in added {
                guard let content = tok["content"] as? String else { continue }
                let id: Int?
                if let i = tok["id"] as? Int {
                    id = i
                } else if let n = tok["id"] as? NSNumber {
                    id = n.intValue
                } else {
                    id = nil
                }
                if let id {
                    specials[content] = id
                    vocab[content] = id
                }
            }
        }

        let padId = specials["<|endoftext|>"] ?? vocab["<|endoftext|>"] ?? 151_643
        let eosId = specials["<|im_end|>"] ?? vocab["<|im_end|>"] ?? 151_645

        return QwenTokenizer(
            vocab: vocab,
            merges: merges,
            specialTokens: specials,
            padTokenId: padId,
            eosTokenId: eosId
        )
    }

    // MARK: - Encode

    /// Encode raw text (no chat template) to token ids.
    public func encode(_ text: String) -> [Int] {
        let pieces = splitOnSpecial(text)
        var ids: [Int] = []
        for piece in pieces {
            if let sid = specialTokens[piece] {
                ids.append(sid)
            } else {
                ids.append(contentsOf: bpeEncode(piece))
            }
        }
        return ids
    }

    /// Apply Klein chat template, encode, pad/truncate to `maxLength` (default 512).
    public func encodePrompt(
        _ prompt: String,
        maxLength: Int = ModelConstants.maxSequenceLength,
        enableThinking: Bool = false
    ) -> (ids: [Int], attentionMask: [Int]) {
        let formatted = QwenChatTemplate.format(
            userPrompt: prompt,
            enableThinking: enableThinking,
            addGenerationPrompt: true
        )
        var ids = encode(formatted)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        var mask = Array(repeating: 1, count: ids.count)
        if ids.count < maxLength {
            let pad = maxLength - ids.count
            ids.append(contentsOf: Array(repeating: padTokenId, count: pad))
            mask.append(contentsOf: Array(repeating: 0, count: pad))
        }
        return (ids, mask)
    }

    // MARK: - BPE internals

    private func splitOnSpecial(_ text: String) -> [String] {
        guard !specialTokenSet.isEmpty else { return [text] }
        let sorted = specialTokenSet.sorted { $0.count > $1.count }
        var result: [String] = []
        var remaining = text[text.startIndex...]

        while !remaining.isEmpty {
            var bestRange: Range<String.Index>?
            var bestToken: String?
            for sp in sorted {
                if let r = remaining.range(of: sp) {
                    if bestRange == nil
                        || r.lowerBound < bestRange!.lowerBound
                        || (r.lowerBound == bestRange!.lowerBound && sp.count > bestToken!.count)
                    {
                        bestRange = r
                        bestToken = sp
                    }
                }
            }
            if let bestRange, let bestToken {
                let before = remaining[remaining.startIndex ..< bestRange.lowerBound]
                if !before.isEmpty {
                    result.append(String(before))
                }
                result.append(bestToken)
                remaining = remaining[bestRange.upperBound...]
            } else {
                result.append(String(remaining))
                break
            }
        }
        return result
    }

    private func bpeEncode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        let utf8 = Array(text.utf8)
        var word: [String] = utf8.map { String(byteEncoder[$0]!) }
        word = applyBPE(word)
        return word.compactMap { vocab[$0] }
    }

    private func applyBPE(_ tokens: [String]) -> [String] {
        var word = tokens
        if word.count < 2 { return word }
        while true {
            var minRank = Int.max
            var minIndex: Int?
            for i in 0 ..< (word.count - 1) {
                if let rank = bpeRanks["\(word[i]) \(word[i + 1])"], rank < minRank {
                    minRank = rank
                    minIndex = i
                }
            }
            guard let idx = minIndex else { break }
            var next: [String] = []
            next.reserveCapacity(word.count - 1)
            var i = 0
            while i < word.count {
                if i == idx {
                    next.append(word[i] + word[i + 1])
                    i += 2
                } else {
                    next.append(word[i])
                    i += 1
                }
            }
            word = next
            if word.count < 2 { break }
        }
        return word
    }

    /// GPT-2 / Qwen `bytes_to_unicode` mapping.
    private static func bytesToUnicode() -> ([UInt8: Character], [Character: UInt8]) {
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var encoder: [UInt8: Character] = [:]
        var decoder: [Character: UInt8] = [:]
        for i in 0 ..< bs.count {
            let byte = UInt8(bs[i])
            let scalar = UnicodeScalar(cs[i])!
            let ch = Character(scalar)
            encoder[byte] = ch
            decoder[ch] = byte
        }
        return (encoder, decoder)
    }
}
