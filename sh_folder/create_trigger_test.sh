activate() {
  /home/heerak/workspace/main_venv/bin/activate
}

cd /home/heerak/workspace/universal-triggers

python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 1 --num_trigger_tokens 3
#python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 1 --num_trigger_tokens 4
#python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 1 --num_trigger_tokens 5

#python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 0 --num_trigger_tokens 3
#python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 0 --num_trigger_tokens 4
#python nsmc/create_trigger.py --run_name original --epoch 1 --dataset_label_filter 0 --num_trigger_tokens 5
