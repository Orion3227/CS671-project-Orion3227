#!/usr/bin/env sh
python scripts/dpse/dpse.py --model dpse --savedir SOMEWHERE --shot 5 --way 5 --cdim 32 --zdim 32 --gpu_num SOMETHING |tee somewhere.log
