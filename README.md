We custom yolov8 to be multi-head to do multi-task

To do that, we must:
- reconstruct the architecture of yolov8
- modify forward process
- modify data loader process
- add loss function for classify, modify loss computing

Pros:
- high real-time: about 17fps to get output if using GPU
- easy for training: because it trains both detection task and classify task at the same time

Cons:
- data highly imbalanced causes problem
- computing resources: we train the model on GTX 3060, 6GB Ram so we can only set input size: 320, batchsize: 20

Notes:
- Training with origin train data (15k image): Score on public test is about 19
- Training with duplicate male data, more black data (15k image): Score on public test is about 21

-> This approach shows the promising
