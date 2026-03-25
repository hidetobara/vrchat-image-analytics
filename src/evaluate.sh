#!/bin/bash

python3 evaluate.py --image /app/data/evaluate/Beyond-a-bit.jpg
python3 evaluate.py --image /app/data/evaluate/City-of-stardust.jpg /app/data/evaluate/District-Roboto.jpg /app/data/evaluate/Gion-Street.jpg --text_embeds /app/data/worlds.text.pt
python3 evaluate.py --image /app/data/evaluate/Nostalgic-Winter-Night.jpg --text_embeds /app/data/worlds.text.pt
python3 evaluate.py --image /app/data/evaluate/Olympia.png --text_embeds /app/data/worlds.text.pt
python3 evaluate.py --image /app/data/evaluate/ORGANISM.png --text_embeds /app/data/worlds.text.pt
python3 evaluate.py --image /app/data/evaluate/Poppy-Street.jpg --text_embeds /app/data/worlds.text.pt
python3 evaluate.py --image /app/data/evaluate/Yayoi-Summer-Nights.jpg \
    /app/data/evaluate/Under-the-Rose.jpg \
    /app/data/evaluate/Sunroom.jpg \
    /app/data/evaluate/Starry-Park.jpg \
    /app/data/evaluate/Self-Reflection.jpg \
    /app/data/evaluate/SAKURA-City.jpg \
    /app/data/evaluate/Rose-Port.jpg \
    --text_embeds /app/data/worlds.text.pt
