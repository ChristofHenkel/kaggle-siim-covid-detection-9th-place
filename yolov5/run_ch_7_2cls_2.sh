python train.py --data covid19_fold0_2cls.yaml --weights yolov5s.pt --hyp hyp_run_ch_1b.yaml --img-size 512 --epochs 25 --project=run_ch_7_2cls_2 --name fold0 --device 1

python train.py --data covid19_fold1_2cls.yaml --weights yolov5s.pt --hyp hyp_run_ch_1b.yaml --img-size 512 --epochs 25 --project=run_ch_7_2cls_2 --name fold1 --device 0

python train.py --data covid19_fold2_2cls.yaml --weights yolov5s.pt --hyp hyp_run_ch_1b.yaml --img-size 512 --epochs 25 --project=runs/run_ch_7_2cls_2 --name fold2 --device 1

python train.py --data covid19_fold3_2cls.yaml --weights yolov5s.pt --hyp hyp_run_ch_1b.yaml --img-size 512 --epochs 25 --project=runs/run_ch_7_2cls_2 --name fold3 --device 2

python train.py --data covid19_fold4_2cls.yaml --weights yolov5s.pt --hyp hyp_run_ch_1b.yaml --img-size 512 --epochs 25 --project=runs/run_ch_7_2cls_2 --name fold4 --device 3
