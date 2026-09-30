#!/usr/bin/env bash
# Place the downloaded datasets where the CSVs expect them, then verify every image by sha256.
#
# Usage (from the repository root); pass only the archives you have, e.g. BEAF alone is enough
# for Tables 1-2 and Figures 1-2:
#     bash scripts/prepare_data.sh --beaf beaf_dataset_ver1.zip \
#         --coco_val2017 val2017.zip --voc2007 VOCtest_06-Nov-2007.tar \
#         --chexpert /path/to/CheXpert-v1.0-small
#
# Layout produced (all under benchmarks/data/, reachable as data/ through the root symlink):
#     coco/images/val2014/     BEAF originals and object-removed images (Tables 1-2, Figures 1-2, Table 3 training)
#     coco/images/val2017/     COCO MCQ (Table 3)
#     voc2007/VOCdevkit/...    VOC2007 MCQ (Table 3)
#     CheXpert-v1.0-small/     CheXpert MCQ (Table 3)
set -euo pipefail
cd "$(dirname "$0")/.."
D=benchmarks/data
BEAF="" COCO="" VOC="" CHEX=""
while [ $# -gt 0 ]; do
    case "$1" in
    --beaf) BEAF="$2"; shift 2 ;;
    --coco_val2017) COCO="$2"; shift 2 ;;
    --voc2007) VOC="$2"; shift 2 ;;
    --chexpert) CHEX="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

verify() {  # $1 = manifest name
    echo "verifying $1 ..."
    sha256sum --quiet -c "$D/checksums/$1.sha256" && echo "  $1: all $(wc -l < "$D/checksums/$1.sha256") images match"
}

if [ -n "$BEAF" ]; then
    mkdir -p $D/coco/images/val2014
    unzip -q -o -j "$BEAF" 'beaf_dataset_ver1/*.png' 'beaf_dataset_ver1/*.jpg' -d $D/coco/images/val2014
    verify beaf
fi
if [ -n "$COCO" ]; then
    mkdir -p $D/coco/images
    unzip -q -o "$COCO" -d $D/coco/images
    verify coco_val2017
fi
if [ -n "$VOC" ]; then
    mkdir -p $D/voc2007
    tar -xf "$VOC" -C $D/voc2007
    verify voc2007
fi
if [ -n "$CHEX" ]; then
    ln -sfn "$(cd "$CHEX" && pwd)" $D/CheXpert-v1.0-small
    verify chexpert
fi
