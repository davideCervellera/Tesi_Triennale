import CoreML

class URLClassifier {
    private let tokenizer: WordPieceTokenizer
    private let model: TinyBertURLClassifier

    /// Inizializzo modello e il vobabolario
    
    init() throws {
        guard let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "txt") else {
            throw NSError(domain: "URLClassifier", code: -1, userInfo: [NSLocalizedDescriptionKey: "vocab.txt not found"])
        }
        self.tokenizer = try WordPieceTokenizer(vocabFilePath: vocabPath)
        self.model = try TinyBertURLClassifier(configuration: MLModelConfiguration())
    }
    
    /// Prende i logits e li trasforma in probabilità, questo mi permette di interpretare meglio il risultato
    
    private func softmax(_ logits: MLMultiArray) -> [Double] {
        let values = (0..<logits.count).map { logits[$0].doubleValue }
        let maxVal = values.max() ?? 0
        let exps = values.map { exp($0 - maxVal) }
        let sumExps = exps.reduce(0, +)
        return exps.map { $0 / sumExps }
    }
    
    /// Normalizza l'url inserito per coerenza con codice python
    private func normalize(url: String) -> String {
        var cleanedURL = url.lowercased()

        
        if !cleanedURL.hasPrefix("http://") && !cleanedURL.hasPrefix("https://") {
            cleanedURL = "http://" + cleanedURL
        }

        
        guard let parsed = URL(string: cleanedURL), let host = parsed.host else {
            return url.lowercased()
        }

        
        var domain = host
        if domain.hasPrefix("www.") {
            domain.removeFirst("www.".count)
        }

        
        var path = parsed.path
        if path.hasSuffix("/") {
            path.removeLast()
        }

        
        return domain + path
    }
    
    /// Questa funzione normalizza l url (tramite chiamata alla funzione normalize), lo tokenizza e riempie un MultiArray con i token ottenuti
    /// Da tutto in input al modello che restituisce dei logits
    /// Tramite la funzione softmax trasforma i logits ottenuti in probabilità, confronta le probabilità ed effettua una predizione

    func classify(url: String) -> String {
        let normalizedURL = normalize(url: url)
        let (inputIds, attentionMask) = tokenizer.encode(normalizedURL)

        guard let inputIdsArray = try? MLMultiArray(shape: [1, 128], dataType: .int32),
              let attentionMaskArray = try? MLMultiArray(shape: [1, 128], dataType: .int32) else {
            return "Errore nella creazione di MLMultiArray"
        }

        for i in 0..<128 {
            inputIdsArray[[0, NSNumber(value: i)]] = NSNumber(value: inputIds[i])
            attentionMaskArray[[0, NSNumber(value: i)]] = NSNumber(value: attentionMask[i])
        }

        do {
            let input = TinyBertURLClassifierInput(input_ids: inputIdsArray, attention_mask: attentionMaskArray)
            let output = try model.prediction(input: input)
            let logits = output.cast_13

            for i in 0..<logits.count {
                let val = logits[i].doubleValue
                if val.isNaN || val.isInfinite {
                    return "Errore: output modello non valido (NaN/Inf)"
                }
            }

            let probabilities = softmax(logits)
            let probSicuro = probabilities[0]
            let probPhishing = probabilities[1]

            return probPhishing > probSicuro ? "Phishing" : "Sicuro"

        } catch {
            return "Errore modello: \(error.localizedDescription)"
        }
    }
}
