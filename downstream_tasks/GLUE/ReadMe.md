

# Tutorial:
Evaluate SenseBert on the GLUE Benchmark.




## First:
Select the GLUE task you want to test:  
cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb,wnli  

(In this directory)  
Unix:  
export TASK_NAME=mrpc

Windows:  
SET TASK_NAME=mrpc

## Second:
Fine-tune base BERT model (Skip if already fine-tuned model is available):

Unix:

python run_glue.py \  
  --model_name_or_path bert-base-uncased \  
  --task_name \$TASK_NAME \
  --do_train \  
  --do_eval \  
  --max_seq_length 128 \  
  --per_device_train_batch_size 32 \  
  --learning_rate 2e-5 \  
  --num_train_epochs 3 \  
  --output_dir models/\$TASK_NAME/Baseline/ \  
  --logging_steps 10000\  
  --save_steps 10000\  
  --overwrite_output_dir  



Windows:

python run_glue.py ^  
  --model_name_or_path bert-base-uncased ^  
  --task_name "%TASK_NAME%" ^  
  --do_train ^  
  --do_eval ^  
  --max_seq_length 128 ^  
  --per_device_train_batch_size 32 ^  
  --learning_rate 2e-5 ^  
  --num_train_epochs 3 ^  
  --output_dir models/"%TASK_NAME%"/Baseline/ ^  
  --logging_steps 10000 ^  
  --save_steps 10000 ^  
  --overwrite_output_dir    
  
  

  Result on my machine:

  ***** eval metrics *****

epoch                   =        3.0  
eval_accuracy           =     0.8382  
eval_combined_score     =     0.8628  
eval_f1                 =     0.8874  
eval_loss               =     0.4161  
eval_runtime            = 0:00:01.74  
eval_samples            =        408  
eval_samples_per_second =    234.365  
eval_steps_per_second   =     29.296  


  
  
## Third:
Go to the main folder and run 'bert.py' script and fill-in the GLUE task.  
Explanation: Create the SensePolar space for the fine-tuned BERT embeddings of step two.


## Forth:
Re-train the last layer with SensePolar embeddings:  
For the smaller Tasks we recommend to re-run the training with different seeds. E.g. '--seed 42'

Unix:  
python run_glue.py \  
  --model_name_or_path models/\$TASK_NAME/Polar/ \  
  --task_name $TASK_NAME \  
  --do_train \  
  --do_eval \  
  --max_seq_length 128 \  
  --per_device_train_batch_size 32 \  
  --learning_rate 2e-5 \  
  --num_train_epochs 5 \  
  --output_dir models/\$TASK_NAME/Polar/ \  
  --logging_steps 10000\  
  --save_steps 10000\  
  --overwrite_output_dir\  
  --use_polar
  
Windows:  
python run_glue.py ^  
  --model_name_or_path models/"%TASK_NAME%"/Polar/ ^  
  --task_name "%TASK_NAME%" ^  
  --do_train ^  
  --do_eval ^  
  --max_seq_length 128 ^  
  --per_device_train_batch_size 32 ^  
  --learning_rate 2e-5 ^  
  --num_train_epochs 5 ^  
  --output_dir models/"%TASK_NAME%"/Polar/ ^  
  --logging_steps 10000 ^  
  --save_steps 10000 ^  
  --overwrite_output_dir ^  
  --use_polar


  ***** eval metrics *****  
  epoch                   =        5.0    
  eval_accuracy           =     0.8186   
  eval_combined_score     =     0.8489  
  eval_f1                 =     0.8791  
  eval_loss               =      0.568  
  eval_runtime            = 0:00:14.67  
  eval_samples            =        408  
  eval_samples_per_second =     27.809  
  eval_steps_per_second   =      3.476  



## optional Fifth: 
In Eval CUDA OOM (there seems to be a memory leak somewhere in the run_glue script):     

Unix:  
  python run_glue.py \  
  --model_name_or_path models/\$TASK_NAME/Polar/ \  
  --task_name $TASK_NAME \  
  --do_eval \  
  --max_seq_length 128 \  
  --output_dir models/\$TASK_NAME/Polar/ \  
  --logging_steps 10000\  
  --save_steps 10000\  
  --overwrite_output_dir\  
  --use_polar\  
  --no_cuda  

Windows:  
python run_glue.py ^  
  --model_name_or_path models/"%TASK_NAME%"/Polar/ ^  
  --task_name "%TASK_NAME%" ^  
  --do_eval ^  
  --max_seq_length 128 ^  
  --output_dir models/"%TASK_NAME%"/Polar/ ^  
  --logging_steps 10000 ^  
  --save_steps 10000 ^  
  --overwrite_output_dir ^  
  --use_polar ^  
  --no_cuda  
