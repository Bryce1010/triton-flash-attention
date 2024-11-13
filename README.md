# Flash Attention implemented with Triton

Implements the Flash Attention 2 algorithm, based on the code published by OpenAI's team at [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

It also includes some cuda examples as shown in the video.

Install the requirements at `triton/requirements.txt` to launch the Python file. Adjust the `BATCH_SIZE`, `NUM_HEADS`, `SEQ_LEN`, `HEAD_DIM` to make sure your computer doesn't explode.

The *naive* implementation materializes a `SEQ_LEN x SEQ_LEN` tensor, so it may be the bottleneck in running this code. Just disable it and try to push the `SEQ_LEN` of the Flash Attention to the limit supported by your hardware.

Not tested on AMD, so let me know!

