card_arr=(1 2 4 8 10)
bs=64

for(( i=0;i<${#card_arr[@]};i++)) do
  card_num=${card_arr[i]}
  log_path="gpu_logs/bert_large_bs${bs}_card${card_num}.qdrep"
  echo python -m torch.distributed.launch --nproc_per_node ${card_num} --use_env nlp_example.py --batch_size $bs 
  nsys profile  -c cudaProfilerApi -f true --stats true  -o ${log_path} python -m torch.distributed.launch --nproc_per_node ${card_num} --use_env nlp_example_nsys.py --batch_size $bs 
done

