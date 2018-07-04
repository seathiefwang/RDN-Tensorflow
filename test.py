from model import RDN
import scipy.misc
from PIL import Image
from PIL import ImageFilter
import argparse
import data
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/test5")
parser.add_argument("--imgsize",default=128,type=int)
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--globallayers",default=16,type=int)
parser.add_argument("--locallayers",default=8,type=int)
parser.add_argument("--featuresize",default=64,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")

def predict(x, image):
    outputs = network.predict(x)
    scipy.misc.imsave(args.outdir+"/input_"+image,x)
    scipy.misc.imsave(args.outdir+"/output_"+image,outputs[0])

args = parser.parse_args()
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = RDN(down_size,args.globallayers,args.locallayers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
if args.image:
    x = Image.open(args.image).convert('RGB')
    x = data.preprocess(np.array(x))
    predict(x, os.path.basename(args.image))
else:
    for filename in os.listdir(args.dataset):
        x = Image.open(args.dataset+'/'+filename).convert('RGB')
        x = data.preprocess(np.array(x))
        predict(x, filename)

