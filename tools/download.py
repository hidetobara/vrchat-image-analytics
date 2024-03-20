import argparse
import csv
import os
import requests
import time


def load_worlds(path):
    worlds = []
    with open(path, "r") as f:
        for items in csv.reader(f):
            if len(items) < 4 or items[0].startswith("#"):
                continue
            wid = items[0]
            title = items[1]
            image_url = items[2]
            worlds.append({"id": wid,  "title": title, "image_url": image_url})
    return worlds

def download_worlds(worlds):
    for w in worlds:
        download_image(w["image_url"], w["id"])
        time.sleep(1)


def download_image(url, filename, dl_dir="/app/data/images"):
    ua_str = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1"
    headers = {
        'User-Agent': ua_str,
        'content-type': 'image/png'
    }

    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code != 200:
        print("STATUS=", response.status_code, url)
        return

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        print("TYPE=", content_type)
        return

    with open(os.path.join(dl_dir, filename + ".png"), "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's download !")
    parser.add_argument('--worlds', help="infomations of worlds, CSV format")
    parser.add_argument('--limit', type=int, default=10, help="limit")
    args = parser.parse_args()

    if args.worlds:
        worlds = load_worlds(args.worlds)
    if args.limit < len(worlds):
        worlds = worlds[:args.limit]
    download_worlds(worlds)
    
