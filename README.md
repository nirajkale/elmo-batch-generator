# elmo-batch-generator
elmo-batch-generator

## A keras generator that can generate elmo embeddings batch by batch
### features
- Support both output modes & signature in elmo model
- Support for internal caching to speed the training second epoch onwards
- method for warming up the emebddings (due to stateful nature elmo's LSTMs, warmup could be useful as suggested in the paper)
- Support bulk conversion of input & export

