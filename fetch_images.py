import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def fetch(name):
    target = Path(__file__).resolve().parent / 'data/images' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get('https://s3.amazonaws.com/images.cocodataset.org/val2017/' + name, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return len(response.content)


if __name__ == '__main__':
    design = json.loads((Path(__file__).resolve().parent / 'data/design.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        print(f'Downloaded {sum(pool.map(fetch, design["images"]))} bytes')
