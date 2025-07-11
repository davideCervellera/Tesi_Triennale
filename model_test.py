from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from urllib.parse import urlparse, urlunparse


# Questo codice carica il modello precedentemente addestrato e permette di testarlo inserendo da tastiera
# degli URL, ricevendo una predizione.



# Questa funzione prende un URL e lo normalizza, rimuovendo il prefisso 'http://' e 'www'.
# Per effettuare correttamente il parsing viene aggiunto all inizio http se mancante.
# Viene poi effettuato un filtraggio degli url torppo lunghi o non validi, che vengono sostituiti
# con una stringa vuota, che viene poi eliminata successivamente.

def normalize_url(url, max_length=128):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        netloc = parsed.netloc

        if netloc.startswith('www.'):
            netloc = netloc[4:]

        normalized_url = urlunparse((
            '',
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))


        if normalized_url.startswith('//'):
            normalized_url = normalized_url[2:]


        if len(normalized_url) > max_length:
            return ''

        return normalized_url
    except Exception:
        return ''


# Questa funzione richiama la funzione di normalize definita precedentemente. Se tutto va bene
# l'url normalizzato viene passato al tokenizer che lo trasforma in un formato adatto per essere
# preso in input dal modello già addestrato.
# Il modello restituisce i logits che vengono trasformati in probabilità e viene stabilito il risultato


def classify_url(url):
    normalized = normalize_url(url)

    if normalized == '':
        return "URL non valido o troppo lungo", 0.0

    inputs = tokenizer(normalized, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item() * 100
    label = "Phishing" if predicted_class == 1 else "Sicuro"
    return label, confidence


# Carico tokenizer e modello pre-addestrato
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
weights = load_file("trained_model/model.safetensors")
model.load_state_dict(weights)
model.eval()

if __name__ == "__main__":
    while True:
        url = input("Inserisci un URL da valutare (o 'exit' per uscire): ").strip()
        if url.lower() == 'exit':
            print("Uscita dall'applicazione.")
            break
        label, confidence = classify_url(url)

        print(f"URL: {url}")
        print(f"Predizione: {label} ({confidence:.2f}%)\n")