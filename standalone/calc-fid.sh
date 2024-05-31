#!/bin/bash

if [ $# -lt 1 ]; then
    echo "calc-fid.sh PATH-TO-BLENDED-IMAGES"
    exit 1
fi

if [ ! -d data/frll_neutral_front_cropped ]; then
    echo "Original data folder \"data/frll_neutral_front_cropped\" is missing. Download it first (hint: See the Makefile)"
    exit 1
fi

BLENDEDPATH=$1
FIDDIMS=(64 192 768 2048)
FIDOUTDIR=results/fid

mkdir -p $FIDOUTDIR
BASENAME=$(basename -- "$BLENDEDPATH")

for d in ${FIDDIMS[@]}; do
    # Calculating the original images FID
    if [ ! -f $FIDOUTDIR/frll_neutral_front_cropped${d}.npz ]; then
        python -m pytorch_fid --dims ${d} --save-stats data/frll_neutral_front_cropped/ $FIDOUTDIR/frll_neutral_front_cropped${d}.npz
    fi

    if [ ! -f $FIDOUTDIR/nw-diffae_${d}.npz ]; then
        python -m pytorch_fid --dims ${d} --save-stats $BLENDEDPATH $FIDOUTDIR/$BASENAME_${d}.npz
    fi

    echo "DIMS = " ${d}
    python -m pytorch_fid $FIDOUTDIR/frll_neutral_front_cropped${d}.npz $FIDOUTDIR/$BASENAME_${d}.npz
done
