import argparse
import csv
import os
import requests
import time

import util


def download_worlds(worlds):
    done = 0
    for n, w in enumerate(worlds):
        time.sleep(0.01)
        flag = download_image(w["image_url"], w["id"])
        if flag >= 0:
            time.sleep(2.0)
        if flag > 0:
            done += 1
        if n % 100 == 0:
            print(f"[DONE/N]={done}/{n}", flush=True)

def download_image(url, wid, dl_dir="/app/data/images") -> int:
    try:
        box_dir, path = util.get_box_path(wid)
        if os.path.exists(path):
            return -1
        os.makedirs(box_dir, exist_ok=True)

        ua_str = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1"
        headers = {
            'User-Agent': ua_str,
            'content-type': 'image/png'
        }

        response = requests.get(url, headers=headers, allow_redirects=True)
        if response.status_code != 200:
            print("STATUS=", response.status_code, url)
            return 0

        content_type = response.headers["content-type"]
        if 'image' not in content_type:
            print("TYPE=", content_type)
            return 0

        with open(path, "wb") as f:
            f.write(response.content)
        return 1
    except Exception as ex:
        print("ERROR", str(ex), wid)
        print("URL=", url)
        return 0

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
    
# python3 download.py --worlds /app/data/best_worlds.2026.csv --limit 1000 > download.out &
