find ../data/raw/data/ -type f | xargs -P 16 -n 9000 \
    mogrify -resize 25%
