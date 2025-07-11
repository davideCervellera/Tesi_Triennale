import Foundation

/// In questa classe l'url fornito in ingresso viene prima di tutto gestito con un basicTokenizer che gestisce la punteggiatura e separa i token di base,
/// in seguito viene gestito da un WordPieceTokenizer che funziona in maniera molto molto simile al tokenizer originale di tinyBERT.
/// Non è stato possibile usare proprio il tokenizer originale perchè il nostro obiettivo è che l'app funzioni completamente in locale senza dipendenze esterne.



class BasicTokenizer {
    func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var currentToken = ""

        for char in text {
            if char.isLetter || char.isNumber {
                currentToken.append(char)
            } else {
                if !currentToken.isEmpty {
                    tokens.append(currentToken)
                    currentToken = ""
                }
                if !char.isWhitespace {
                    tokens.append(String(char))
                }
            }
        }

        if !currentToken.isEmpty {
            tokens.append(currentToken)
        }

        return tokens
    }
}


class WordPieceTokenizer {
    let vocab: [String: Int]
    let unkToken = "[UNK]"
    let clsToken = "[CLS]"
    let sepToken = "[SEP]"
    let padToken = "[PAD]"
    let maxLength: Int
    let basicTokenizer = BasicTokenizer()

    init(vocabFilePath: String, maxLength: Int = 128) throws {
        self.maxLength = maxLength
        let vocabContent = try String(contentsOfFile: vocabFilePath, encoding: .utf8)
        var tempVocab = [String: Int]()
        let lines = vocabContent.components(separatedBy: .newlines)
        for (index, line) in lines.enumerated() {
            if !line.isEmpty {
                tempVocab[line] = index
            }
        }
        self.vocab = tempVocab
    }

    func tokenize(_ text: String) -> [String] {
        let cleanedText = text.lowercased()
        
        
        let basicTokens = basicTokenizer.tokenize(cleanedText)
        
        var tokens: [String] = []

        for token in basicTokens {
            tokens.append(contentsOf: wordpieceTokenize(token))
        }

        return tokens
    }

    private func wordpieceTokenize(_ word: String) -> [String] {
        var tokens: [String] = []
        var start = word.startIndex

        while start < word.endIndex {
            var end = word.endIndex
            var curSubstr: String? = nil

            while end > start {
                let substr = String(word[start..<end])
                let candidate = (start == word.startIndex) ? substr : "##" + substr

                if vocab.keys.contains(candidate) {
                    curSubstr = candidate
                    break
                }
                end = word.index(before: end)
            }

            if let sub = curSubstr {
                tokens.append(sub)
                start = end
            } else {
                tokens.append(unkToken)
                break
            }
        }

        return tokens
    }

    func encode(_ text: String) -> (inputIds: [Int], attentionMask: [Int]) {
        var tokens = [clsToken]
        tokens += tokenize(text)
        tokens.append(sepToken)

        var inputIds = tokens.map { vocab[$0] ?? vocab[unkToken]! }

        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength))
        } else {
            inputIds += Array(repeating: vocab[padToken]!, count: maxLength - inputIds.count)
        }

        let attentionMask = inputIds.map { $0 == vocab[padToken]! ? 0 : 1 }

        return (inputIds, attentionMask)
    }
}
