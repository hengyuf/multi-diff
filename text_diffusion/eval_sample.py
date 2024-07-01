import os
import math
import time

import imageio
import numpy as np
import torch
import pickle
import argparse
from diffusion_utils.utils import add_parent_path

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=16)
parser.add_argument('--length', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--benchmark', type=eval, default=False)
parser.add_argument('--num_bandit', type=int, default=4)
parser.add_argument('--num_data', type=int, default=16000)
parser.add_argument('--type_data', type=str, default='')

N_list=[2**i*5 for i in range(7,12)]
K_list=[4]
for N in N_list:
    for K in K_list:
        
        eval_args = parser.parse_args()
        assert eval_args.length is not None, 'Currently, length has to be specified.'
        eval_args.model=f'/bicmr/home/hengyuf04/log/flow/bandit/multinomial_diffusion_v2/expdecay/N={N}K={K}'+eval_args.type_data
        #eval_args.num_bandit=K
        #eval_args.num_data=N

        path_args = '{}/args.pickle'.format(eval_args.model)
        path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

        torch.manual_seed(eval_args.seed)

        ###############
        ## Load args ##
        ###############

        with open(path_args, 'rb') as f:
            args = pickle.load(f)

        ##################
        ## Specify data ##
        ##################
        #args.type_data="_state_action"

        train_loader, eval_loader, data_shape, num_classes = get_data(args)

        ###################
        ## Specify model ##
        ###################
        print(f"num of classes:{num_classes}")
        if eval_args.type_data=='_state_action':
            num_classes=num_classes*(2**num_classes)

        model = get_model(args, data_shape=data_shape, num_classes=num_classes)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)
        checkpoint = torch.load(path_check,map_location=device)
        model.load_state_dict(checkpoint['model'])
        print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

        ############
        ## Sample ##
        ############

        path_samples = os.path.join('/bicmr/home/hengyuf04/Diffusion/multinomial_diffusion2/text_diffusion', f'samples/sample_N{N}_K{K}'+eval_args.type_data+'new.txt')
        path_samples_chain = os.path.join('/bicmr/home/hengyuf04/Diffusion/multinomial_diffusion2/text_diffusion', f'samples/sample_chain_N{N}_K{K}'+eval_args.type_data+'new.txt')
        if not os.path.exists(os.path.dirname(path_samples)):
            os.mkdir(os.path.dirname(path_samples))

        
        model = model.to(device)
        model = model.eval()
        if eval_args.double: model = model.double()

        lengths = torch.ones(eval_args.samples, device=device, dtype=torch.long) * eval_args.length
        # mask = length_mask(lengths, maxlen=data_shape[0])

        if eval_args.benchmark:
            torch.cuda.synchronize()
            results = []
            with torch.no_grad():
                for _ in range(10):
                    start = time.time()
                    out = model.sample(eval_args.samples)
                    torch.cuda.synchronize()
                    results.append(time.time() - start)
            print()
            print(f'Sample time average {np.mean(results):.2f} +/- {np.std(results):.2f}')
            quit()

        samples_chain = model.sample_chain(eval_args.samples).cpu()
        samples = samples_chain[0]

        # samples = model.sample(eval_args.samples)
        torch.save(samples,path_samples)
        torch.save(samples_chain,path_samples_chain)




def chain_linspace(samples_chain_text, num_steps=150, repeat_last=10):
    out = []
    for i in np.linspace(0, len(samples_chain_text)-1, num_steps):
        idx = int(i)
        if idx >= len(samples_chain_text):
            print('index too big')
            idx = idx - 1
        out.append(samples_chain_text[idx])

    for i in range(repeat_last):
        out.append(samples_chain_text[-1])
    return out


def format_text(batch_text):
    # print('batch_text', batch_text)
    out = []
    for text in batch_text:
        linesize = 90
        reformat = text[0:linesize]
        for i in range(linesize, len(text), linesize):
            reformat = reformat + '\n' + text[i:i+linesize]

        out.append(reformat)

        # print('reformat', reformat)

    return '\n\n'.join(out)


def draw_text_to_image(text, invert_color=False):
    font = ImageFont.truetype("CourierPrime-Regular.ttf", 24)

    black = (0, 0, 0)
    white = (255, 255, 255)
    if invert_color:
        background_color = white
        textcolor = black
    else:
        background_color = black
        textcolor = white

    img = Image.new('RGB', (1290, 200), color=background_color)

    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (10, 10), text, textcolor, font=font)

    img_np = np.array(img)
    return img_np

