import argparse
import csv
import os
import requests
import time

import util


def download_worlds(worlds):
    for n, w in enumerate(worlds):
        if download_image(w["image_url"], w["id"]) == 0:
            continue
        time.sleep(1)
        if n % 100 == 0:
            print(f"[DONE] N={n}")

def download_image(url, filename, dl_dir="/app/data/images") -> int:
    path = os.path.join(dl_dir, filename + ".png")
    if os.path.exists(path):
        print("ALREADY=", filename)
        return 0

    ua_str = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1"
    headers = {
        'User-Agent': ua_str,
        'content-type': 'image/png'
    }

    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code != 200:
        print("STATUS=", response.status_code, url)
        return -1

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        print("TYPE=", content_type)
        return -1

    with open(os.path.join(dl_dir, filename + ".png"), "wb") as f:
        f.write(response.content)
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's download !")
    parser.add_argument('--worlds', help="infomations of worlds, CSV format")
    parser.add_argument('--limit', type=int, default=10, help="limit")
    args = parser.parse_args()

    if args.worlds:
        worlds = util.load_worlds(args.worlds)
    if args.limit < len(worlds):
        worlds = worlds[:args.limit]
    download_worlds(worlds)
    
