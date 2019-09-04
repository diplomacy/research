# Supervised and RL Models for No Press Diplomacy

This repository contains the source code used to develop a supervised and RL agent that can play the No Press version of Diplomacy.  

<p align="center">
  <img width="500" src="docs/images/map_overview.png" alt="Diplomacy Map Overview">
</p>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Restrictions: The trained weights provided with this repository are for research purposes only and cannot be used to power any bots on any website without my prior written consent, which may be withheld without reasons.**

**The data provider also prevents using its data to train any bots accessible on any website.**

**You can play against the trained model by playing against "KestasBot" on webdiplomacy.net**

## Dataset

The model was trained by using a dataset of 156,468 games (diplomacy-v1-27k-msgs.zip), which consists of:

- 16,633 games on non-standard maps (e.g. modern and ancmed) (other_maps.jsonl)
- 33,279 no-press games on the standard map (standard_no_press.jsonl)
- 50 press games on the standard map with messages (standard_press_with_msgs.jsonl)
- 106,456 press games on the standard map without messages (standard_press_without_msgs.jsonl)
- 50 public press games on the standard map with messages (standard_public_press.jsonl)

A dataset of 156,458 games with 13,469,536 messages is also being prepared, but it is not yet available.

Access to the dataset used to train the model can be requested by sending an email to webdipmod@gmail.com.


## Getting Started

### Installation

The repository can be installed in a conda environment with:

```python3
conda create -n diplomacy
conda activate diplomacy
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

This package depends on Redis and singularity 3+. Singularity can be installed with:

```bash
# Installing Singularity v3.2.0
export VERSION=v3.2.0
sudo apt-get update -y
sudo apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev libseccomp-dev pkg-config squashfs-tools

# Installing GO 1.12.5
export GO_VERSION=1.12.5 OS=linux ARCH=amd64
wget -nv https://dl.google.com/go/go$GO_VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzf go$GO_VERSION.$OS-$ARCH.tar.gz
rm -f go$GO_VERSION.$OS-$ARCH.tar.gz
export GOPATH=$HOME/.go
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin
mkdir -p $GOPATH
go get github.com/golang/dep/cmd/dep

# Building from source
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs
git clone https://github.com/sylabs/singularity.git
cd singularity
git checkout $VERSION
./mconfig -p /usr/local
cd ./builddir
make
sudo make install
```

The package is compatible with Python 3.5, 3.6, and 3.7.

### Training models

To train a model:

```bash
$ export WORKING_DIR=/path/to/some/directory
$ cp diplomacy-v1-27k-msgs.zip $WORKING_DIR
$ conda activate diplomacy
$ python diplomacy_research/scripts/build_dataset.py
$ python diplomacy_research/models/policy/order_based/train.py --model_id 12
```

### Playing against the SL and RL agents

It is possible to play against the published results by using the `DipNetSLPlayer` and `DipNetRLPlayer` players in `diplomacy_research.players.benchmark_player`.

These players will automatically download a singularity container with the trained weights, and then launch a TF serving server to handle the requests.

A simple example on how to play a 7 bots game is:

```python3
from tornado import gen
import ujson as json
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.players.benchmark_player import DipNetSLPlayer
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop

@gen.coroutine
def main():
    """ Plays a local game with 7 bots """
    player = DipNetSLPlayer()
    game = Game()

    # Playing game
    while not game.is_game_done:
        orders = yield {power_name: player.get_orders(game, power_name) for power_name in game.powers}
        for power_name, power_orders in orders.items():
            game.set_orders(power_name, power_orders)
        game.process()

    # Saving to disk
    with open('game.json', 'w') as file:
        file.write(json.dumps(to_saved_game_format(game)))
    stop_io_loop()

if __name__ == '__main__':
    start_io_loop(main)
```

### Playing against a model

It is also possible for humans to play against bots using the web interface. The player can be changed in `diplomacy_research.scripts.launch_bot`

```bash
# In a terminal window or tab - Launch React server (from diplomacy/diplomacy)
npm start

# In another terminal window or tab - Launch diplomacy server
python -m diplomacy.server.run

# In a third terminal window or tab - Launch the bot script
python diplomacy_research/scripts/launch_bot.py
```

### Trained weights and experiment logs

To facilitate reproducibility, the experiments can be downloaded using the following links. These include hyperparameters, tensorboard graphs, output logs, and weights for each epoch.

- Order based LSTM model (order-based v12 - Accuracy of 61.3% - **DipNet SL**) [Download - 5.4GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/order-based-lstm.zip)
- Order based Transformer model (order-based v15 - Accuracy of 60.7%) [Download - 8.2GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/order-based-trsf.zip)
- Token based LSTM model (token-based v10 - Accuracy of 60.3%) [Download - 6.0GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/token-based-lstm.zip)
- Token based Transformer model (token-based v11 - Accuracy of 58.9%) [Download - 3.5GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/token-based-trsf.zip)
- RL Model (Bootstrapped from order-based v12 and value v1 - **DipNet RL**) [Download - 11.1GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/rl-model.zip)

### Games against Albert (DAIDE)

The 1v6 and 6v1 games played between DipNet SL and Albert (DAIDE) can be downloaded below:

- List of games with power assignments [Download - 53KB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/daide_albert_results.xlsx)
- Visualisation of each game (svg and json) [Download - 2.3GB](https://f002.backblazeb2.com/file/ppaquette-public/benchmarks/experiments/daide_albert_games.zip)
