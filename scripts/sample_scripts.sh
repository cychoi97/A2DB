#!/bin/bash

DATASET_DIR="/workspace/changyong/04.PECT_Enhancement/data/CCY_PE_DECT"
SRC_LIST=(src_80keV src_100keV src_125keV)
TRG_LIST=(trg trg trg)
MASTER_PORT="6020"
SBAE_CKPT="pect_sbae_A6000_Input_LambdaReg1e-5"
LDM_CKPT="pect_sbae_A6000_Input_LambdaReg1e-5_SpatialRescaler"
LOAD_ITR=50000
LDM_LOAD_ITR=250000
NFE1=2
NFE2=10
NFE3=100

python sample.py \
	--dataset-dir ${DATASET_DIR} \
    --src ${SRC_LIST[0]} \
    --trg ${TRG_LIST[0]} \
	--master-port ${MASTER_PORT} \
	--sbae-ckpt ${SBAE_CKPT} \
    --ldm-ckpt ${LDM_CKPT} \
    --load-itr ${LOAD_ITR} \
    --ldm-load-itr ${LDM_LOAD_ITR} \
	--nfe ${NFE1} \
	--clip-denoise \
    --save-dicom \
	--use-fp16

python sample.py \
	--dataset-dir ${DATASET_DIR} \
    --src ${SRC_LIST[1]} \
    --trg ${TRG_LIST[1]} \
	--master-port ${MASTER_PORT} \
	--sbae-ckpt ${SBAE_CKPT} \
    --ldm-ckpt ${LDM_CKPT} \
    --load-itr ${LOAD_ITR} \
    --ldm-load-itr ${LDM_LOAD_ITR} \
	--nfe ${NFE1} \
	--clip-denoise \
    --save-dicom \
	--use-fp16

python sample.py \
	--dataset-dir ${DATASET_DIR} \
    --src ${SRC_LIST[2]} \
    --trg ${TRG_LIST[2]} \
	--master-port ${MASTER_PORT} \
	--sbae-ckpt ${SBAE_CKPT} \
    --ldm-ckpt ${LDM_CKPT} \
    --load-itr ${LOAD_ITR} \
    --ldm-load-itr ${LDM_LOAD_ITR} \
	--nfe ${NFE1} \
	--clip-denoise \
    --save-dicom \
	--use-fp16
