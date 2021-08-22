set -e

for FOLD in {0..4}
do
python train.py --fold $FOLD "$@"
done
