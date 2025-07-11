import pandas as pd
from urllib.parse import urlparse, urlunparse


# Questa funzione prende un URL e lo normalizza, rimuovendo il prefisso 'http://' e 'www'.
# Per effettuare correttamente il parsing viene aggiunto all inizio http se mancante.
# Viene poi effettuato un filtraggio degli url torppo lunghi o non validi, che vengono sostituiti
# con una stringa vuota, che viene poi eliminata successivamente.

def normalize_url(url, max_length=128):
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc

        if not netloc:
            netloc = parsed.path
            parsed = urlparse("http://" + url)
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




input_csv = 'dataset_urls.csv'
output_csv = 'normalized_dataset_urls.csv'

# Leggo il file csv, applico la normalizzazione chiamando la relativa funzione, filtro via le righe con url vuoti
# rimuovo i duplicati e salvo in un nuovo file

df = pd.read_csv(input_csv)
df['url'] = df['url'].apply(normalize_url)
df = df[df['url'] != '']
df = df.drop_duplicates(subset='url', keep='first')
df.to_csv(output_csv, index=False)

print(f"URL normalizzati, duplicati rimossi e salvati in '{output_csv}'")