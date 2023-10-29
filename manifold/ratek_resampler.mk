# ratek_resampler.mk
#
# Makefile to build up curves for various VQ schemes
#
# usage:
#
# 1. Use train_120.spc (runs fairly quickly, good for initial tests)
#
#   cd codec2/build/linux
#   make -f ../script/ratek_resampler.mk
#
# 2. Use train.spc 
#
#    TRAIN=train M=4096 make -f ../script/ratek_resampler.mk
#
# 3. Generate .tex/.eps file for use in report
#
#  epslatex_path=../doc/ratek_resampler/ TRAIN=train M=4096 make -f ../script/ratek_resampler.mk

TRAIN ?= train_120
M ?= 512

SHELL  := /bin/bash
CODEC2 := $(HOME)/codec2
TRAIN_FULL := ~/Downloads/$(TRAIN).spc

PLOT_DATA := $(TRAIN)_k20_res1.txt $(TRAIN)_k80_res1.txt
	      
all: $(TRAIN)_ratek.png

# always re-render PNGs
FORCE:

$(TRAIN)_ratek.png: $(PLOT_DATA) FORCE
	echo "ratek_resampler_plot(\"$(TRAIN)_ratek.png\", \
	         \"$(TRAIN)_k20_res1.txt\",'b-*;k20 1;', \
			 'continue',\"$(TRAIN)_k20_res2.txt\",'b-*;k20 2;', \
 			 'continue',\"$(TRAIN)_k20_res3.txt\",'b-*;k20 3;', \
             \"$(TRAIN)_k80_res1.txt\",'g-+;k80 1;', \
             'continue', \"$(TRAIN)_k80_res2.txt\",'g-+;k80 2;', \
             'continue', \"$(TRAIN)_k80_res3.txt\",'g-+;k80 3;' \
             ); quit" | octave-cli -p $(CODEC2)/octave --no-init-file

# (1) K=20 two stage VQ
$(TRAIN)_k20_res1.txt: $(TRAIN)_b20.f32
	K=20 Kst=0 Ken=19 M=$(M) stage3='yes' ./ratek_resampler.sh train_lbg $(TRAIN)_b20.f32 $(TRAIN)_k20

# (1) K=80 two stage VQ
$(TRAIN)_k80_res1.txt: $(TRAIN)_y80.f32
	K=79 Kst=0 Ken=78 M=$(M) stage3='yes' ./ratek_resampler.sh train_lbg $(TRAIN)_y80.f32 $(TRAIN)_k80

$(TRAIN)_b20.f32:
	./ratek_resampler.sh gen_train_b $(TRAIN_FULL) $(TRAIN)_b20.f32

$(TRAIN)_y80.f32:
	./ratek_resampler.sh gen_train_y $(TRAIN_FULL) $(TRAIN)_y80.f32

clean:
	rm -f $(PLOT_DATA)
	