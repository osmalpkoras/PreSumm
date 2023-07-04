```bash
cd src

python train.py  -task abs -mode train -bert_data_path ../bert_data -dec_dropout 0.2  -model_path ../checkpoints/BertAbs/cnndm -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1 -log_file ../logs/abs_bert_cnndm

# train BertAbs
python train.py  -task abs -mode train -bert_data_path ../bert_data/cnndm -dec_dropout 0.2  -model_path ../checkpoints/BertAbs/cnndm -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 5  -log_file ../logs/abs_bert_cnndm

# evaluate
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path ../bert_data/cnndm -log_file ../logs/val_abs_bert_cnndm -model_path ../checkpoints/BertAbs/cnndm -sep_optim true -use_interval true -visible_gpus 5 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm 

# train Transformer baseline
python train.py -mode train -accum_count 5 -batch_size 300 -bert_data_path ../bert_data/cnndm -dec_dropout 0.1 -log_file ../logs/cnndm_baseline -lr 0.05 -model_path ../checkpoints/baseline/cnndm -save_checkpoint_steps 2000 -seed 777 -sep_optim false -train_steps 200000 -use_bert_emb true -use_interval true -warmup_steps 8000  -visible_gpus 5 -max_pos 512 -report_every 50 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -enc_dropout 0.1 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048 -encoder baseline -task abs
```
