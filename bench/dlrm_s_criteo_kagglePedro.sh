#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python dlrm_s_pytorch.py"
#dlrm_c2_bin="python dlrm_s_caffe2.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
#$dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
#$dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/trainday0day0.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --save-model=dlrm_criteo_kaggle.pytorch $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
#python dlrm_s_pytorch.py \
#--arch-sparse-feature-size=16 \
#--arch-mlp-bot="13-512-256-64-16" \
#--arch-mlp-top="512-256-1" \
#--data-generation=dataset \
#--data-set=kaggle \
#--raw-data-file=./input/trainday0day0.txt \
#--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
#--loss-function=bce \
#--round-targets=True \
#--learning-rate=0.1 \
#--mini-batch-size=128 \
#--print-freq=256 \
#--test-freq=128 \
#--enable-profiling \
#--mlperf-logging \
#--plot-compute-graph \
#--print-time \
#--test-mini-batch-size=256 \
#--test-num-workers=16 \
#--save-model=dlrm_criteo_kaggle.pytorch \
#$dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
#--debug-mode \
#python dlrm_s_pytorch.py \
#--arch-sparse-feature-size=16 \
#--arch-mlp-bot="13-512-256-64-16" \
#--arch-mlp-top="512-256-1" \
#--data-generation=dataset \
#--data-set=kaggle \
#--raw-data-file=./input/trainday0day0day0.txt \
#--processed-data-file=./input/kaggleAdDisplayChallenge_processed2.npz \
#--loss-function=bce \
#--round-targets=True \
#--learning-rate=0.1 \
#--mini-batch-size=128 \
#--print-freq=128 \
#--test-freq=128 \
#--enable-profiling \
#--mlperf-logging \
#--plot-compute-graph \
#--print-time \
#--test-mini-batch-size=128 \
#--test-num-workers=16 \
#--save-model=dlrm_criteo_kaggle.pytorch \
#$dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
#python dlrm_s_pytorch.py \
#--arch-sparse-feature-size=16 \
#--arch-mlp-bot="13-512-256-64-16" \
#--arch-mlp-top="512-256-1" \
#--data-generation=dataset \
#--data-set=kaggle \
#--raw-data-file=./input/trainday0day0day0day0.txt \
#--processed-data-file=./input/kaggleAdDisplayChallenge_processed3.npz \
#--loss-function=bce \
#--round-targets=True \
#--learning-rate=0.1 \
#--mini-batch-size=32 \
#--print-freq=64 \
#--test-freq=64 \
#--enable-profiling \
#--mlperf-logging \
#--plot-compute-graph \
#--print-time \
#--test-mini-batch-size=32 \
#--test-num-workers=16 \
#--save-model=dlrm_criteo_kaggle.pytorch \
#$dlrm_extra_option 2>&1 | tee run_kaggle_pt.log
python dlrm_s_pytorch.py \
--arch-sparse-feature-size=18 \
--arch-mlp-bot="13-143-411-18" \
--arch-mlp-top="296-1" \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/trainday0day0day0day0.txt \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed3.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.09999214850893416 \
--mini-batch-size=32 \
--print-freq=64 \
--test-freq=64 \
--enable-profiling \
--mlperf-logging \
--plot-compute-graph \
--print-time \
--test-mini-batch-size=32 \
--test-num-workers=16 \
--save-model=dlrm_criteo_kaggle.pytorch \
$dlrm_extra_option 2>&1 | tee run_kaggle_pt.log


#echo "run caffe2 ..."
## WARNING: the following parameters will be set based on the data set
## --arch-embedding-size=... (sparse feature sizes)
## --arch-mlp-bot=... (the input to the first layer of bottom mlp)
##$dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log
#$dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/test.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

# Visualisation
# Connand line examples:
#
# Full analysis of embeddings and data representations for Criteo Kaggle data:
# $python ./tools/visualize.py --data-set=kaggle --load-model=../dlrm-2020-05-25/criteo.pytorch-e-0-i-110591
#         --raw-data-file=../../criteo/input/train.txt --skip-categorical-analysis
#         --processed-data-file=../../criteo/input/kaggleAdDisplayChallenge_processed.npz
#
#
# To run just the analysis of categoricala data for Criteo Kaggle data set:
# $python ./tools/visualize.py --data-set=kaggle --load-model=../dlrm-2020-05-25/criteo.pytorch-e-0-i-110591 \
#         --raw-data-file=../../criteo/input/train.txt --data-randomize=none --processed-data-file=../../criteo/input/kaggleAdDisplayChallenge_processed.npz \
#         --skip-embedding --skip-data-plots
#
# $python visualize.py --data-set=kaggle --load-model=dlrm_criteo_kaggle.pytorch --raw-data-file=input/trainday0day0.txt --skip-categorical-analysis --processed-data-file=input/kaggleAdDisplayChallenge_processed.npz
#
# $python visualize.py --data-set=kaggle --raw-data-file=input/trainday0day0.txt --skip-categorical-analysis --processed-data-file=input/kaggleAdDisplayChallenge_processed.npz

echo "done"
