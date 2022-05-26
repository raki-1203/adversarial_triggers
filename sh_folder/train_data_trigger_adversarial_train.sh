activate() {
  /home/heerak/workspace/main_venv/bin/activate
}

cd /home/heerak/workspace/universal-triggers

# origin + adv(rate) & trigger 개수 테스트
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.1 --num_trigger 1 --trigger_type train_data --plus_data True
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.2 --num_trigger 1 --trigger_type train_data --plus_data True
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.3 --num_trigger 1 --trigger_type train_data --plus_data True
python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.4 --num_trigger 1 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 1 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 1 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 1 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 1 --trigger_type train_data --plus_data True

#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 2 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 2 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 2 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 2 --trigger_type train_data --plus_data True

#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.5 --num_trigger 3 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.6 --num_trigger 3 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.7 --num_trigger 3 --trigger_type train_data --plus_data True
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train --rate 0.8 --num_trigger 3 --trigger_type train_data --plus_data True

# origin(1-rate) + adv(rate) & trigger 1개 테스트
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train-same_number_of_data --rate 0.1 --num_trigger 1 --trigger_type train_data
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train-same_number_of_data --rate 0.2 --num_trigger 1 --trigger_type train_data
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train-same_number_of_data --rate 0.3 --num_trigger 1 --trigger_type train_data
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train-same_number_of_data --rate 0.4 --num_trigger 1 --trigger_type train_data
#python nsmc/nsmc_adversarial_training.py --run_name trigger_adv_train-same_number_of_data --rate 0.5 --num_trigger 1 --trigger_type train_data
