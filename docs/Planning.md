# Streaming Transformer Project Planning

## Project description

TODO

## Planning 

### Models

* [ ] Implement WER, RTF metrics calculation
* [ ] Train only embeddings of WavTokens
* [ ] Train embeddings of WavTokens & fine-tune LLM 
* [ ] Add streaming to model processing: add audio by chunks and 
make model predict tokens of added chunk. During WavTokens calculation 
try different approaches of WavTokens calculation: chunks of different size, 
overlapping chunks, non-overlapping chunks and etc. 
* [ ] Add speculative decoding to streaming model (train/fine-tune/distill smaller model)
* [ ] Add attention window sliding
* [ ] Try to drop already processed WavTokens from input in order to reduce context size in attention
* [ ] ? Add/research triggers handling ? 

### Server

* [ ] Implement gRPC server stand for latency and load measuring
* [ ] Implement client that makes queries to the server and calculates overall processing latency 
* [ ] Add batching

## Datasets 

* LibriSpeech
* Datasets from [Whisper paper](https://arxiv.org/abs/2212.04356)

## Metrics
* Word Error Rate
* Alignment Error Rate 
* Length Error Rate
* Real time factor