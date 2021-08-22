python train.py --data covid19_fold0_05.yaml --weights yolov5x.pt --hyp hyp_run_ch_2b.yaml --img-size 512 --epochs 20 --project=run_ch_2b --name fold0 --device 0

python train.py --data covid19_fold1_05.yaml --weights yolov5x.pt --hyp hyp_run_ch_2b.yaml --img-size 512 --epochs 20 --project=run_ch_2b --name fold1 --device 1

python train.py --data covid19_fold2_05.yaml --weights yolov5x.pt --hyp hyp_run_ch_2b.yaml --img-size 512 --epochs 20 --project=run_ch_2b --name fold2 --device 2

python train.py --data covid19_fold3_05.yaml --weights yolov5x.pt --hyp hyp_run_ch_2b.yaml --img-size 512 --epochs 20 --project=run_ch_2b --name fold3 --device 3

python train.py --data covid19_fold4_05.yaml --weights yolov5x.pt --hyp hyp_run_ch_2b.yaml --img-size 512 --epochs 20 --project=run_ch_2b --name fold4 --device 0
