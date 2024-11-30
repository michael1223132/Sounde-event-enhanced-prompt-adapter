export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python inference.py --original_args="/data4/xiongchenxu/tango/tango/pretrained/main_config.json" --model="/data4/xiongchenxu/tango/tango/saved/1725003033/best_model.bin" \
--test_file="/data4/xiongchenxu/tango/tango/data/test_v2.json" --test_references="/data4/xiongchenxu/tango/tango/target_audio" \
--num_steps 200 --guidance 3 --num_samples 1 --batch_size 4 > inference_output_ada.log 2>&1 &