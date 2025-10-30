for l in 150 300 500 1000; do 
    # for training 
    python encode.py -i ./train_example/tr/host_tr.fa -l $l -p host
    python encode.py -i ./train_example/tr/virus_tr.fa -l $l -p virus
    # for validation
    python encode.py -i ./train_example/val/host_val.fa -l $l -p host
    python encode.py -i ./train_example/val/virus_val.fa -l $l -p virus
done

# Training multiple models for different contig lengths
for l in 150 300 500 1000; do 

    python training.py -l $l -i ./train_example/tr/encoded -j ./train_example/val/encoded -o ./train_example/new_models -f 10 -n 500 -d 500 -e 10
done