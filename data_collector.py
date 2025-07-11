import requests
import pandas as pd
import io


# Scarico la lista dei siti di phishing e la lista di siti famosi
# Per fare in modo di avere un dataset equilibrato prelevo i primi 50.000 URL in entrambi i casi


# Scarico gli URL di phishing da phishtank.com , che mi fornise un file json

def download_phishing_data():
    url = 'https://data.phishtank.com/data/online-valid.json'
    response = requests.get(url)
    response.raise_for_status()

    phishtank_json = response.json()

    urls = []
    for entry in phishtank_json:
        url_str = entry.get('url')
        if url_str:
            urls.append(url_str)

    df = pd.DataFrame({'url': urls})
    df['label'] = 'phishing'
    return df.head(50000)


# Scarico gli URL dei siti famosi da majestic.com, che mi fornisce un file csv

def download_secure_data():
    url = 'https://downloads.majestic.com/majestic_million.csv'
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df_secure = df[['Domain']].copy()
    df_secure.rename(columns={'Domain': 'url'}, inplace=True)
    df_secure['label'] = 'sicuro'

    return df_secure.head(50000)



# Creo un unico file csv che contiene tutti gli url, sia phishing che sicuri

def main():
    print("Scaricando dati malevoli...")
    df_malware = download_phishing_data()
    print(f"Dati malevoli scaricati: {len(df_malware)} URL")

    print("Scaricando dati sicuri...")
    df_secure = download_secure_data()
    print(f"Dati sicuri scaricati: {len(df_secure)} URL")

    df = pd.concat([df_malware, df_secure], ignore_index=True)
    df.to_csv('dataset_urls.csv', index=False)
    print(f"Dataset unificato creato con successo! Totale URL: {len(df)}")


if __name__ == "__main__":
    main()
