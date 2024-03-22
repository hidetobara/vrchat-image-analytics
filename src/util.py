import csv

def load_worlds(path="/app/data/best_worlds.csv"):
    worlds = []
    with open(path, "r") as f:
        for items in csv.reader(f):
            if len(items) < 5 or items[0].startswith("#"):
                continue
            wid = items[0]
            author = items[1]
            title = items[2]
            image_url = items[4]
            worlds.append({"id": wid, "author": author, "title": title, "image_url": image_url})
    return worlds