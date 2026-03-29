import argparse
import glob
import os
import shutil


def move(old_dir: str, new_dir: str):
    for i, path in enumerate(glob.glob(os.path.join(old_dir, "**/*.png"), recursive=True)):
        wid = os.path.splitext(os.path.basename(path))[0]
        header = wid[-2:]
        to_dir = os.path.join(new_dir, header)
        os.makedirs(to_dir, exist_ok=True)
        to = os.path.join(to_dir, wid + ".png")
        if os.path.exists(to):
            continue
        shutil.copy(path, to)
        print(i, "DONE=", to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's move old to new !")
    parser.add_argument('--src', help="old folder")
    parser.add_argument('--count', action="store_true", help="count")
    parser.add_argument('--to', help="new folder")
    args = parser.parse_args()

    if args.src:
        if args.count:
            print("COUNT=", len(glob.glob(os.path.join(args.src, "**/*.png"), recursive=True)))
        elif args.to:
            move(args.src, args.to)

