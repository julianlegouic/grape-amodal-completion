#!/bin/bash

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.1 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.1" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.2 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.2" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.3 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.3" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.4 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.4" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.5 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.5" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.6 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.6" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.7 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.7" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.8 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.8" &&

python -W ignore -m samples.coco.grape evaluate --dataset='./data/fastgrape/grape/val/' --model=last --amodal --amodal_model=baseline_0.9 >> res.txt &&
echo '-----------------------------' >> res.txt && echo "Done with baseline 0.9";
