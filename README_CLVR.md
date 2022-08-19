# CLVR Implementation Project

Author: Jiahang(Tak) Li

## ToDo

* Save learning rate changes into logs
* Modify value function predictor to output exponential
* Evaluate trained agent with live critic information
* Brain split the backbone into 2 independent parts (black box debugging)
* Reason if it's problem with LSTM setup
  + Stateful batching is error-prone, but it should be working now
* Print the "advantage" values and the predicted ones, see if it makes sense
  + Well... no, cuz we're facing the ep-40 cutoff problem
  + But with the bootstrap fix, it should be correct now
* Print the gradient distribution from actor and critic, and see if it matches
  + If not, we may need to tune learning rate
  + We can also set learning rate for backbone separately
  + But we should start with learning rate grid search
* Reason if it's better to freeze backbone for one but not the other
* Reason if we need to fine-tune PPO hyperparams

## Done
* Visualize the trained policy and see if it makes sense
  + With sufficient training steps, state-based agent is doing what's expected
* Learn to predict the reward offline and show that the model is capable of learning it
  + Yes, the model can predict; see `encoder_dist`
* Try to run more experiments on Google Colab with GPU (folder `gpu_scripts`)
  + Yes, scripts now automatically make use of GPU
  + Pipeline shown to be working
* PPO itself is proven to be working, by `ppo_state_v0`
  + Frozen backbone + PPO is proven to be working, by `ppo_pretrained_image_v0`


## Environment Setup

I trained encoders on MacBook CPU, and PPO on Nvidia 3080 (CUDA Windows).
* CPU: ~1.2 seconds per batch of 64 x steps of 40
* GPU: ~30 seconds per epoch of 4000 steps (100 trajectories)

```bash
conda create --name spinningup python=3.6
git clone git@github.com:TakLee96/clvr_impl_starter.git
git clone git@github.com:openai/spinningup.git

# change setup.py torch version
cd spinningup
pip3 install -e .

cd ../clvr_impl_starter
pip3 install -r requirements.txt
```

To train encoder:
```bash
. scripts/train_encoder.sh
```

To train ppo experiment:
```bash
. scripts/train_ppo_pretrained_image_gamma0_v0.sh
```


## File Structure

These are the new files:
```
logs/            -- logs of all training results
scripts/         -- bash scripts to launch experiments (hold parameters)
spinup/
  ppo.py         -- ppo training loop
  core.py        -- actor/critic definition
  plot.py        -- plot reward x step curve
  test_policy.py -- test policy and save trajectory as images
main.py          -- python entrypoint (uses fire.Fire)
model.py         -- encoder/decoder definition
train.py         -- encoder/decoder training loop
README_CLVR.md   -- this markdown file
```



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

```
RewardPredictor(
  (image_encoder): ImageEncoder(
    (conv_layers): Sequential(
      (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(2, 2))
      (1): ReLU()
      (2): Conv2d(4, 8, kernel_size=(3, 3), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2))
      (5): ReLU()
      (6): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
      (7): ReLU()
      (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
      (9): ReLU()
    )
    (projection): Linear(in_features=64, out_features=64, bias=True)
  )
  (lstm_predictor): LSTMPredictor(
    (lstm): LSTM(64, 64, batch_first=True)
    (mlpk): ModuleDict(
      (dist): MLP(
        (mlp): Sequential(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=1, bias=True)
        )
      )
    )
  )
)
```

By the way, it seems that RAdam is not quite helping with encoder training.

### Doubts about Fully Convolutional Model

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


### Stateful Encoder

The image encoder is stateful rather than stateless, because the LSTM wants to know
the encoding of the previous frame, so that it updates its embedding accordingly;
I'm left with 2 hack-y solutions:

* Inject LSTM embedding into state representation, OR
* Make `actor_critic.step()` function stateful, requiring `.reset()` when episode ends

I ended up going with the 2nd approach as I think it is cleaner; however, the code
would not work, if the environment we're working with can terminate early; the
batching during actor and critic gradient update part will be messed up. For those
environments, we would need to pad trajectories and use done masks.


### Non-converging Finetune

It appears that if I naively train actor and critic with shared encoder in end-to-end
fashion, the training process does not improve reward over long period of time.


### Doubts about Critic

Because our environment has fixed length, I think the value function of each state
not only depends on current state but also remaining time left. It's very weird to force
the model to learn this kind of value. And it's insane that the algorithm is still
working anyways.



