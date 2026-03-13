# Experiment A (ok, need to finish post-implementation):
## task:
Modify from FPL/jsc/train_run_baseline.py, create a new script to train the model using baseline approach beta decrease following the strategy that from max to 0 when epochs finished.
with following configs:
1. Cernbox dataset, epochs 20000
2. OpenML dataset, epochs 20000
3. Cernbox dataset, epochs 200000
4. OpenML dataset, epochs 200000
output: 
same as FPL2/jsc/run_all.py
with input paras we can decide which experiment to run.
also pick out the best checkpoints itself and accuracy near targeted ebops and store it
(JSC OpenML target 191, 1994, 2512, 7285, 7200, 15066, 74847)
(JSC CERNBox target 1385, 3143, 9796, 9801, 10512, 42813, 44537, 46744, 50277, 69034, 109922, 297274)
# Experiment B (TBD):
## task:
1. Train NN for HLF JSC OpenML and then target 191, 1994, 2512, 7285, 7200, 15066, 74847 with our training approaches
2. Train NN for HLF JSC CERNBox and then target 1385, 3143, 9796, 9801, 10512, 42813, 44537, 46744, 50277, 69034, 109922, 297274
output:
same as FPL2/jsc/run_all.py
also pick out the best checkpoints itself and accuracy near targeted ebops and store it
(JSC OpenML target 191, 1994, 2512, 7285, 7200, 15066, 74847)
(JSC CERNBox target 1385, 3143, 9796, 9801, 10512, 42813, 44537, 46744, 50277, 69034, 109922, 297274)
# Experiment C (ok):
Use
- Random init
- Magnitude
- Sensitivity
- Synflow
- Grasp
- SNIP
- Porp.
to prune and train the model with proposed approach at 
- 400
- 1551
- 2585
- 6839
- 11718
ebops
find Max Accuracy around EBOPs
# Experiment D (aborted):
Floor value
- 0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
to prune and train the model with proposed approach at 
- 400
- 1551
- 2585
- 6839
- 11718
ebops
find Max Accuracy around EBOPs
# Experiment E (ok):
find minimux epochs for target ebops 400 to decide how many epochs to give in experiment 1
# Experiment G (ok):
find best beta_restart_decay
show beta scheduler's contri
# Experiment E-> Experiment G -> Experiment A -> Experiment B -> Experiment C