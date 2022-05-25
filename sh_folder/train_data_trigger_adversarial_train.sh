activate() {
  /home/heerak/workspace/main_venv/bin/activate
}

cd /home/heerak/workspace/universal-triggers

python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 1 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 1 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 1 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 1 --trigger_type train_data

python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 2 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 2 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 2 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 2 --trigger_type train_data

python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 3 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 3 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 3 --trigger_type train_data
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 3 --trigger_type train_data
