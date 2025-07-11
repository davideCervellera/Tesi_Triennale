import SwiftUI

struct ContentView: View {
    @State private var urlText = ""
    @State private var result = ""
    @State private var classifier: URLClassifier?
    @State private var isLoading = false
    @State private var errorMessage: String?

    /// Elementi relativi all'interfaccia grafica
    var body: some View {
        VStack(spacing: 30) {
            header

            TextField("Inserisci URL", text: $urlText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .padding(.horizontal)

            classifyButton

            if isLoading {
                ProgressView("Analisi in corso...")
                    .padding()
            }

            if let error = errorMessage {
                Text("⚠️ \(error)")
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            } else if !result.isEmpty {
                Text(result)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(resultColor)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Spacer()
        }
        .padding()
        .onAppear {
            do {
                classifier = try URLClassifier()
            } catch {
                errorMessage = "Errore caricamento modello: \(error.localizedDescription)"
            }
        }
    }

    /// Colore del risultato in base al tipo di classificazione
    private var resultColor: Color {
        if result.contains("Sicuro") {
            return .green
        } else if result.contains("Phishing") {
            return .red
        } else {
            return .primary
        }
    }

    /// Intestazione dell'app
    private var header: some View {
        Text("🔒 SafeURL")
            .font(.largeTitle.bold())
            .foregroundColor(.accentColor)
    }

    /// Pulsante per avviare la classificazione
    private var classifyButton: some View {
        Button(action: classifyURL) {
            HStack {
                Image(systemName: "checkmark.shield")
                Text("Classifica").bold()
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.accentColor)
            .foregroundColor(.white)
            .cornerRadius(12)
            .shadow(radius: 5)
        }
        .disabled(urlText.isEmpty || isLoading)
        .padding(.horizontal)
    }

    /// Classifica l'URL inserito dall'utente
    private func classifyURL() {
        guard urlText.count > 6 else {
            errorMessage = "URL troppo corto per una classificazione significativa."
            result = ""
            return
        }

        guard let classifier = classifier else {
            errorMessage = "Modello non disponibile."
            result = ""
            return
        }

        errorMessage = nil
        result = ""
        isLoading = true

        DispatchQueue.global(qos: .userInitiated).async {
            let prediction = classifier.classify(url: urlText)

            DispatchQueue.main.async {
                isLoading = false
                if prediction.hasPrefix("Errore") {
                    errorMessage = prediction
                } else {
                    result = prediction
                }
            }
        }
    }
}
