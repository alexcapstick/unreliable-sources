




python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-no_c-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name no_c --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-no_c-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name no_c --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-no_c-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name no_c --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-no_c-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name no_c --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-no_c-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name no_c --num-rounds 25 --batch-size 128






cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_cs-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_cs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_cs-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_cs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_cs-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_cs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_cs-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_cs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_cs-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_cs --num-rounds 25 --batch-size 128




cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_rl-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_rl --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_rl-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_rl --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_rl-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_rl --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_rl-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_rl --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_rl-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_rl --num-rounds 25 --batch-size 128




cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_no-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_no --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_no-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_no --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_no-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_no --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_no-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_no --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_no-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_no --num-rounds 25 --batch-size 128




cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbf-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_lbf --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbf-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_lbf --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbf-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_lbf --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbf-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_lbf --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbf-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_lbf --num-rounds 25 --batch-size 128




cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_ns-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_ns --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_ns-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_ns --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_ns-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_ns --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_ns-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_ns --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_ns-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_ns --num-rounds 25 --batch-size 128




cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbs-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 1 --corruption-name c_lbs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbs-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 2 --corruption-name c_lbs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbs-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 3 --corruption-name c_lbs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbs-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 4 --corruption-name c_lbs --num-rounds 25 --batch-size 128

cd ../../
python benchmark_data_save.py --dataset-name cifar10 --model-name Conv3Net-c_lbs-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 4 --source-size 128 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset cifar10 --model cnn -t medium --eval-every 1 --seed 5 --corruption-name c_lbs --num-rounds 25 --batch-size 128
