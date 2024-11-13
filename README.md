# Flash Attention implemented with Triton

Implements the Flash Attention 2 algorithm, based on the code published by OpenAI's team at [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

It also includes some cuda examples as shown in the video.

Install the requirements at `triton/requirements.txt` to launch the Python file. Adjust the `BATCH_SIZE`, `NUM_HEADS`, `SEQ_LEN`, `HEAD_DIM` to make sure your computer doesn't explode.

The *naive* implementation materializes a `SEQ_LEN x SEQ_LEN` tensor, so it may be the bottleneck in running this code. Just disable it and try to push the `SEQ_LEN` of the Flash Attention to the limit supported by your hardware.

Not tested on AMD, so let me know!

## Exercise 1: autotuning the backwards pass

Can you apply autotuning configs to the backwards pass like done for the forward pass?

## Exercise 2: how to make Flash Attention faster

As you can see, during the backwards pass we are going through the entire `SEQ_LEN` even when the attention calculation is `causal`, can you avoid going through all tokens that would not contribute to any change in `dK`, `dQ` and `dV` when the attention calculation is causal?