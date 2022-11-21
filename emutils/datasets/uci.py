import time
import re
import os
import pandas as pd
import urllib.request, urllib.parse
import bs4 as bs
import numpy as np
from tqdm import tqdm

from emutils import PACKAGE_DATA_FOLDER

__all__ = [
    'get_uci_dataframe',
    'download_uci_dataset',
]

UCI_DATASETS_URL = "https://archive.ics.uci.edu/ml/datasets.php"
UCI_DATASET_BASE_URL = "https://archive.ics.uci.edu/ml/"


def get_data_folder_url(dataset_page_url):
    soup = bs.BeautifulSoup(urllib.request.urlopen(dataset_page_url).read(), 'lxml')

    # Let's get the link to the data folder
    for link in soup.find_all("a"):
        # The first link that we found is the data folder
        if link["href"].find("machine-learning-databases") != -1:
            return urllib.parse.urljoin(dataset_page_url, link["href"])

    # Nothing found
    return None


def get_data_files_url_from_data_page(data_page_url):
    soup = bs.BeautifulSoup(urllib.request.urlopen(data_page_url).read(), 'lxml')
    data_urls = [urllib.parse.urljoin(data_page_url, a['href']) for a in soup.findAll('a')]
    data_urls = [file_url for file_url in data_urls if not file_url.endswith('Index') and not file_url.endswith('/')]
    return data_urls


def get_uci_dataframe(data_directory=PACKAGE_DATA_FOLDER, cache_days=np.inf, cache_filename='uci.csv', sleep_time=.5):
    # Create data directory
    os.makedirs(data_directory, exist_ok=True)

    # Cache filename
    cache_filename = os.path.join(PACKAGE_DATA_FOLDER, cache_filename)

    # Re-download
    if not os.path.exists(cache_filename) or (time.time() - os.path.getmtime(cache_filename)) / 86400 > cache_days:
        # Download (urllib) & Parse (BS4)
        soup = bs.BeautifulSoup(urllib.request.urlopen(UCI_DATASETS_URL).read(), 'lxml')

        # Extract the table
        tables = soup.find_all('table')
        tables = [table for table in tables if len(table) > 100]
        assert len(tables) == 1, "There should be only one table this large (n > 1000). The one with the datasets."
        table = [tr.findAll('td') for tr in tables[0].findAll('tr')]
        table_header, table_rows = table[0], [tr for tr in table[1:] if len(tr) > 2]

        # # Extract columns
        columns = ["".join(td.findAll(text=True)) for td in table_header]

        # # Extract text
        data = [["".join(td.findAll(text=True)).strip() for td in tr][2:] for tr in table_rows]

        # Create a dataframe
        df = pd.DataFrame(data, columns=columns)

        # return table_rows

        # Extract dataset links
        df['dataset_page'] = [
            urllib.parse.urljoin(UCI_DATASET_BASE_URL, tr[0].findAll('a')[0]['href']) for tr in table_rows
        ]

        # Cache
        df.to_csv(cache_filename, index=False)

    else:
        # Load cache
        df = pd.read_csv(cache_filename)
        if 'data_urls' in df:
            df['data_urls'] = df['data_urls'].apply(lambda x: [s[1:-1] for s in x[1:-1].split(',')]
                                                    if isinstance(x, str) else None)

    # Extract data folder links
    if 'data_page' not in df:
        data_folder_links = []
        for url in tqdm(df['dataset_page'].values, desc='Extracting UCI Data Folders URLs'):
            time.sleep(sleep_time)
            try:
                data_folder_links.append(get_data_folder_url(url) if url is not None else None)
            except:
                data_folder_links.append(None)
        df['data_page'] = data_folder_links

        # Cache
        df.to_csv(cache_filename, index=False)

    # Extract data urls
    if 'data_urls' not in df:
        data_urls = []
        for url in tqdm(df['data_page'].values, desc='Extracting Data Pages URLs'):
            time.sleep(sleep_time)
            try:
                data_urls.append(get_data_files_url_from_data_page(url) if url is not None else None)
            except:
                data_urls.append(None)
        df['data_urls'] = data_urls

        # Save cache
        df.to_csv(cache_filename, index=False)

    # Clean the data_directory name
    if 'data_directory' not in df:

        def clean_name(name):
            name = re.sub(r"\s", "_", name)
            name = name.lower()
            if len(name) > 30:
                name = "_".join(name[:30].split('_')[:-1])
            return name

        df['data_directory'] = df['Name'].apply(clean_name)

    return df


def download_uci_dataset(dataset_name, **kwargs):

    # Get datasets DataFrame
    df = get_uci_dataframe(**kwargs)
    df = df.set_index('Name')

    # Select
    dataset = df.loc[dataset_name]

    # Create directory
    dataset_directory = os.path.join(PACKAGE_DATA_FOLDER, dataset['data_directory'])
    os.makedirs(dataset_directory, exist_ok=True)

    data_files = []

    # If we have data URLs
    if dataset['data_urls'] is not None:
        for url in tqdm(dataset['data_urls'], desc=f'Downloading {dataset_name} data'):

            # Filename where to save
            filename = os.path.join(dataset_directory, os.path.basename(urllib.parse.urlparse(url).path))
            data_files.append(filename)

            # Download and save (if necessary)
            if not os.path.exists(filename):
                with open(filename, 'wb') as f:
                    f.write(urllib.request.urlopen(url).read())
    else:
        print('No data URLs for this dataset.')

    return dataset_directory