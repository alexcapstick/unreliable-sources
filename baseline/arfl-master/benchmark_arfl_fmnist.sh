




python benchmark_data_save.py --dataset-name fmnist --model-name MLP-no_c-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name no_c --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-no_c-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name no_c --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-no_c-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name no_c --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-no_c-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name no_c --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-no_c-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name no_c --num-rounds 40 --batch-size 200






cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_cs-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_cs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_cs-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_cs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_cs-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_cs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_cs-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_cs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_cs-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_cs --num-rounds 40 --batch-size 200




cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_rl-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_rl --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_rl-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_rl --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_rl-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_rl --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_rl-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_rl --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_rl-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_rl --num-rounds 40 --batch-size 200




cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_no-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_no --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_no-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_no --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_no-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_no --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_no-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_no --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_no-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_no --num-rounds 40 --batch-size 200




cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbf-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_lbf --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbf-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_lbf --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbf-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_lbf --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbf-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_lbf --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbf-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_lbf --num-rounds 40 --batch-size 200




cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_ns-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_ns --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_ns-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_ns --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_ns-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_ns --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_ns-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_ns --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_ns-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_ns --num-rounds 40 --batch-size 200




cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbs-drstd --seed 1 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 1 --corruption-name c_lbs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbs-drstd --seed 2 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 2 --corruption-name c_lbs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbs-drstd --seed 3 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 3 --corruption-name c_lbs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbs-drstd --seed 4 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 4 --corruption-name c_lbs --num-rounds 40 --batch-size 200

cd ../../
python benchmark_data_save.py --dataset-name fmnist --model-name MLP-c_lbs-drstd --seed 5 -v --n-sources 10 --n-corrupt-sources 6 --source-size 200 --test-method traditional
cd ./arfl-master/models/
python main.py --dataset fmnist --model mlp -t medium --eval-every 1 --seed 5 --corruption-name c_lbs --num-rounds 40 --batch-size 200
