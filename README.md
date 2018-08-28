# lunarDQfD
This repo contains the scripts needed to generate the results found in [Lunar Landings from Demonstrations](https://medium.com/p/27b553b00ce2/edit).

### Usage
##### Install
1. `git clone https://github.com/jakegrigsby/lunarDQfD.git`

##### Watch Finished Agents
1. `python agents.py --model (student or expert) --mode test`

##### Run Main Experiment
1. `python agents.py --model expert --mode train`
2. `python agents.py --model expert --mode demonstrate`
3. `python agents.py --model student --mode train`

#### Dependencies
1. Keras
2. Numpy
3. keras-rl
  * note that the master version of keras-rl may be missing my additions. See [my fork](https://github.com/jakegrigsby/keras-rl).
