card_arr=(1 2 4 8 10)
bs=64

for(( i=0;i<${#card_arr[@]};i++)) do
  card_num=${card_arr[i]}
  echo python -m torch.distributed.launch --nproc_per_node ${card_num} --use_env nlp_example.py --batch_size $bs 
  python -m torch.distributed.launch --nproc_per_node ${card_num} --use_env nlp_example.py --batch_size $bs 
done

