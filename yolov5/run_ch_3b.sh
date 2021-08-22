python train.py --data covid19_fold0_05.yaml --weights yolov5m.pt --hyp hyp_run_ch_3b.yaml --img-size 640 --epochs 20 --project=run_ch_3b --name fold0 --device 0

python train.py --data covid19_fold1_05.yaml --weights yolov5m.pt --hyp hyp_run_ch_3b.yaml --img-size 640 --epochs 20 --project=run_ch_3b --name fold1 --device 1

python train.py --data covid19_fold2_05.yaml --weights yolov5m.pt --hyp hyp_run_ch_3b.yaml --img-size 640 --epochs 20 --project=run_ch_3b --name fold2 --device 2

python train.py --data covid19_fold3_05.yaml --weights yolov5m.pt --hyp hyp_run_ch_3b.yaml --img-size 640 --epochs 20 --project=run_ch_3b --name fold3 --device 3

python train.py --data covid19_fold4_05.yaml --weights yolov5m.pt --hyp hyp_run_ch_3b.yaml --img-size 640 --epochs 20 --project=run_ch_3b --name fold4 --device 0
