
import argparse
from algos import ALGOS


ap = argparse.ArgumentParser()
ap.add_argument('--config_file', type=str, required=True)
ap.add_argument('--expt_name', type=str, required=True)
ap.add_argument('--model', type=str, required=True)
ap.add_argument('--algo', type=str, required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--load', type=str, default=None)
ap.add_argument('--wandb', action='store_true', default=False)
ap.add_argument('--batch_size', type=int, default=100)
ap.add_argument('--num_workers', type=int, default=4)
ap.add_argument('--weight_decay', type=float, default=1e-05)
ap.add_argument('--train_epochs', type=int, default=100)
args = ap.parse_args()
    
# Run the main function
assert args.algo in ALGOS, f'Unrecognized algorithm {args.algo}. Available: {list(ALGOS.keys())}'
ALGOS[args.algo](args)
