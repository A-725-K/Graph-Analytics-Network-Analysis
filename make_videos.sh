#!/bin/bash

VID_DIR='imgs/attack/videos'
OUT_DIR="$VID_DIR/mp4/"
ATTACKS='random closeness betweenness hits pagerank clustering'


function err_exit() {
	RED='\033[0;31m'
	echo -e "${RED}Usage: ./make_videos.sh [--small]"
	exit 1
}


# checking the cli arguments
if [ $# -gt 1 ]; then
	err_exit
elif [ $# -eq 1 ] && [[ "$1" != '--small' ]]; then
	err_exit
fi

# small random graph or realistic graph
pre=''
if [ $# == 1 ] && [ $1 == "--small" ]; then
	pre="small"
else
	pre="big"
fi

echo "--------------------------- $pre"

# creating the videos using ffmpeg
for atk in $ATTACKS; do
	how_many=$(ls -1 $VID_DIR | grep $pre | grep $atk | wc -l)
	echo "$atk --- $how_many"
	
	r=0
	if [ $how_many -gt 50 ]; then
		r=1
	else
		r=2
	fi

	ffmpeg -r $r -i "$VID_DIR/${pre}_$atk%04d.png" -c:v libx264 -vf 'fps=60,format=yuv420p' "$OUT_DIR${pre}_$atk.mp4"
done