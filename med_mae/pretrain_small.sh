CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_small_patch16_dec512d2b \
 --mask_ratio 0.90 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random' \
 --random_resize_range 0.5 1.0 \
 --datasets_names chexpert chestxray_nih
