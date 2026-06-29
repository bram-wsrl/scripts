import time
import json
import requests
import pandas as pd


def request_details(parameter, location_code):
    '''
        Extract the thresholds from the details sidebar on
        waterinfo.rws.nl website for a given parameter and location code.
    '''
    url_fmt = fr'https://waterinfo.rws.nl/api/chart/get?mapType={parameter}&locationCodes={location_code}&values=-48%2C48'

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    try:
        r = requests.get(url_fmt, headers=headers)
        if r.status_code != 200:
            raise requests.exceptions.RequestException(f"Request failed with status code {r.status_code}")
        return r
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e} for parameter: {parameter} and location_code: {location_code}")
        return None


def parse_limits_to_csv(data):
    '''
        Parse the limits from the JSON response and convert them into a pandas DataFrame.
    '''
    limits = data.get('limits', [])
    csv_limits = {}
    for limit in limits:
        for key, value in limit.items():
            csv_limits.setdefault(key, []).append(value)

    df = pd.DataFrame(csv_limits)
    return df


if __name__ == '__main__':
    parameter = 'waterhoogte'
    location_codes = [
        'megen.maas', 'mook', 'grave.boven', 'grave.beneden',
        'lith.boven', 'lith.sluis', 'hank.bergschemaas', 'heesbeen',
        'werkendam.nieuwemerwede', 'zaltbommel', 'tiel.waal', 'dodewaard',
        'nijmegen.waal', 'westervoort.ijsselkop',
        'arnhem.nederrijn', 'driel.beneden', 'driel.boven', 
        'amerongen.boven', 'amerongen.beneden', 'culemborg',
        'hagestein.boven', 'hagestein.beneden',
        'krimpenaandelek.lek', 'schoonhoven', 'dordrecht.oudemaas.benedenmerwede'
    ]

    dfs = []
    for location_code in location_codes:
        time.sleep(0.5)
        response = request_details(parameter, location_code)

        if response:
            data = response.json()
            df = parse_limits_to_csv(data)
            df.insert(0, 'location_code', location_code)
            df['to'] = df['to'].astype(float) / 100
            df['from'] = df['from'].astype(float) / 100
            dfs.append(df)

    dfinal = pd.concat(dfs, ignore_index=True)
    dfinal.to_csv('limits.csv', index=False, encoding='utf-8', sep=';')

    dfs = [df[~df['isNormal']].drop(['softColor', 'color'], axis=1).transpose() for df in dfs]
    pd.concat(dfs).to_csv('limits_filtered.csv', index=False, encoding='utf-8', sep=';')

    # example
    if response:
        with open('example-response.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(response.json()))
