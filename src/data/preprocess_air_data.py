import numpy as np
import pandas as pd
from lxml import etree as ET

def preprocess_air_data():
    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    print(f"Version: {root.attrib['verzija']}")
    print(f"Source: {root.find('vir').text}")
    print(f"Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"Preparation Date: {root.find('datum_priprave').text}")

    station_codes = set(tree.xpath('//postaja/@sifra'))

    for sifra in station_codes:
        postaja_elements = tree.xpath(f'//postaja[@sifra="{sifra}"]')

        columns = ["Date_to", "PM10", "PM2.5"]
        df = pd.DataFrame(columns=columns)

        for postaja in postaja_elements:
            date_to = postaja.find('datum_do').text
            pm10 = postaja.find('pm10').text if postaja.find('pm10') is not None else np.nan
            pm2_5 = postaja.find('pm2.5').text if postaja.find('pm2.5') is not None else np.nan

            df = pd.concat([df, pd.DataFrame([[date_to, pm10, pm2_5]], columns=columns)], ignore_index=True)

        df = df.replace("", np.nan)
        df = df.replace("<1", 1)
        df = df.replace("<2", 2)
        df["Date_to"] = pd.to_datetime(df["Date_to"], errors='coerce')
        df = df.sort_values(by="Date_to")

        csv_path = f"data/preprocessed/air/{sifra}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
if __name__ == "__main__":
    preprocess_air_data()
    