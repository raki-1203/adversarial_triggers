activate() {
  /home/heerak/workspace/main_venv/bin/activate
}

cd /home/heerak/workspace/universal-triggers

# inference test data set
python nsmc/inference.py --run_name train_data-trigger_adv_train-0.1-1
python nsmc/inference.py --run_name train_data-trigger_adv_train-0.2-1
python nsmc/inference.py --run_name train_data-trigger_adv_train-0.3-1
python nsmc/inference.py --run_name train_data-trigger_adv_train-0.4-1
