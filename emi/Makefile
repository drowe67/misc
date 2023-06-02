# generate plots

all:
	cat measurement.txt | grep " 7.1 " | tr -s ' ' | cut -f4 -d' ' > forty.txt
	cat measurement.txt | grep " 14.2 " | tr -s ' ' | cut -f4 -d' ' > twenty.txt
	echo "measurement" | octave-cli -qf
	sed -i 's/fontsize{5}/fontsize{8}/g' measurement.tex
	sed -i 's/fontsize{4}/fontsize{8}/g' measurement.tex

