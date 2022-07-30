# CLVR Implementation Project

Author: Jiahang(Tak) Li

## Problems Encountered

### Description of Encoder

The image that described the encoder is a bit vague on the "dependency on the past"
part; where it illustrates 3 past frames being encoded by a "MLP" and fed into LSTM
as initialization. I took it in my own interpretation and designed the network to:

* Generate CNN embeddings independently for each frame of input
* Project the CNN embedding into LSTM input dimension (without 3-layer MLP)
* Treat them as the input sequence to the LSTM
* LSTM is an encoder only structure that directly connects with MLP to output
  rewards for each time step

### Stateful Encoder

The image encoder is stateful rather than stateless, because the LSTM wants to know
the encoding of the previous frame, so that it updates its embedding accordingly;
I'm left with 2 hack-y solutions:

* Inject LSTM embedding into state representation, OR
* Make `actor_critic.step()` function stateful, requiring `.reset()` when episode ends

I ended up going with the 2nd approach as I think it is cleaner; however, the code
would not work, if the environment we're working with can terminate early; the
batching during actor and critic gradient update part will be messed up.


## Future Work

In theory, the convolution operation is translation invariant, meaning that if we
keep running the convolution operation until the image size goes to 1x1, then we
completely lose location information; we're left with a representation that tells
us whether there is a triangle or a square in the original image without information
about its original position.

Using this convolution backbone to train for a reward function that depends on
horizontal and vertical location of the "agents" is forcing the network to learn
something it is not intended to learn. Even thought the model is powerful enough
that it actually converged, I still think it's likely learning weird hacks, e.g:

* Use LSTM to figure out motion pattern and guess agent locations
* Use the fact that convolution padding always =0 to develop its sense of border
  and thus become capable to compute x and y offsets from border
