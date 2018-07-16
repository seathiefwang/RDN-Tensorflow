import data
import argparse
from model import RDN
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--testset",default="")
parser.add_argument("--imgsize",default=128,type=int)
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--globallayers",default=16,type=int)
parser.add_argument("--locallayers",default=8,type=int)
parser.add_argument("--featuresize",default=64,type=int)
parser.add_argument("--batchsize",default=16,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=10000,type=int)
parser.add_argument("--usepre",default=0,type=int)
args = parser.parse_args()
data.load_dataset(args.dataset,args.testset,args.imgsize)
down_size = args.imgsize//args.scale
network = RDN(down_size,args.globallayers,args.locallayers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,args.scale))
network.train(args.iterations,args.savedir,args.usepre)
