#!/bin/bash
# PLEASE change all "./zh_en_data" to the path where your data is stored

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=train.zh --train-tgt=train.en --dev-src=dev.zh --dev-tgt=dev.en --vocab=vocab_zh_en.json --cuda --lr=5e-4 --patience=1 --valid-niter=1000 --batch-size=32 --dropout=.25
elif [ "$1" = "test" ]; then
	if [ "$2" = "" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin test.zh test.en outputs/test_outputs.txt --cuda
	else
		CUDA_VISIBLE_DEVICES=0 python run.py decode $2 test.zh test.en outputs/test_outputs.txt --cuda
	fi
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=train.zh --train-tgt=train.en --dev-src=dev.zh --dev-tgt=dev.en --vocab=vocab_zh_en.json --lr=5e-4
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin test.zh test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=train.zh --train-tgt=train.en vocab_zh_en.json		
else
	echo "Invalid Option Selected"
fi