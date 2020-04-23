#!/bin/bash

VID_DIR='imgs/attack/videos'
OUT_DIR="$VID_DIR/mp4/"
ATTACKS='random closeness betweenness hits pagerank clustering'

rm -rf $OUT_DIR
mkdir $OUT_DIR

for atk in $ATTACKS; do
	how_many=$(ls -1 $VID_DIR | grep $atk | wc -l)
	echo "$atk --- $how_many"
	
	r=0
	if [ $howmany -gt 50 ]; then
		r=1
	else
		r=2
	fi

	ffmpeg -r $r -i "$VID_DIR/$atk%04d.png" -c:v libx264 -vf 'fps=60,format=yuv420p' "$OUT_DIR$atk.mp4"
done
