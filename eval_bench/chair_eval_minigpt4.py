import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision.transforms import v2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add Nullu to path for minigpt4
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes to ONLY/
nullu_path = os.path.abspath(os.path.join(base_dir, 'Nullu'))
if nullu_path not in sys.path:
    sys.path.insert(0, nullu_path)
sys.path.append(os.path.join(nullu_path, 'minigpt4'))

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, StoppingCriteriaSub
from transformers import StoppingCriteriaList

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils import dist_util
from utils.logger import create_logger
from chair_loader import CHAIRDataset

from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings(action='ignore')

import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on MiniGPT-4 with ONLY.")
    parser.add_argument("--model_path", type=str, help="path to MiniGPT-4 checkpoint")
    parser.add_argument("--model_base", type=str, default="minigpt")
    
    parser.add_argument("--cfg_path", type=str, default="./Nullu/minigpt4/minigpt4_llama2_eval.yaml")
    parser.add_argument("--llama_model_path", type=str, help="path to Llama-2-7b-chat-hf model")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="./logs/chair")
    parser.add_argument("--out_path", type=str, default="./chair_results/minigpt", help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--use_ritual", type=str2bool, default=False)
    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--method_name", type=str, default='none')
    parser.add_argument("--enhance_layer_index", type=int, default=0)

    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.1)

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = "minigpt4"
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing MiniGPT-4 Model')

    # Load MiniGPT-4 model
    cfg = Config(args)
    if args.llama_model_path:
        cfg.model_cfg.llama_model = args.llama_model_path
    
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda')
    
    if args.model_path:
        logger.info(f"Loading MiniGPT-4 checkpoint from {args.model_path}")
        ckpt = torch.load(args.model_path, map_location="cpu")
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    tokenizer = model.llama_tokenizer
    
    logger.info(f"Load llama-2-7b-chat-hf from: {cfg.model_cfg.llama_model}")

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=vis_processor,
        model="minigpt"
    )
    chair_loader = DataLoader(
        chair_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    os.makedirs(args.out_path, exist_ok=True)

    # ==============================================
    #               Augmentations
    # ==============================================
    aug_dict = {
        'horizontal flip': v2.RandomHorizontalFlip(p=1),
        'vertical flip': v2.RandomVerticalFlip(p=1),
        'rotation': v2.RandomRotation(degrees=180),
        'color jitter': v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
        'gaussian blur': v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
        'crop': v2.RandomResizedCrop(size=224),
    }
    pos_aug_counter = {k: 0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        if batch_id == args.num_eval_samples:
            break
            
        img_id = data["image_id"]
        image_path = data["image_path"][0]
        image = data["image"]

        qs = "Please describe this image in detail."

        image_pos = None
        image_neg = None
        
        if args.use_ritual:
            raw_image = Image.open(image_path)
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = vis_processor(raw_image_pos).unsqueeze(0)
                image_pos = torch.tensor(image_pos)
            pos_aug_counter[pos_aug] += 1
            logger.info(f"RITUAL Transformation: {pos_aug}")
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)

        # Prepare image
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processor(raw_image).unsqueeze(0).to('cuda')
        
        # Encode image
        img_embeds, atts_img = model.encode_img(image_tensor)
        img_list = [img_embeds]
        
        # Prepare prompt
        prompt = qs
        prompt_segs = prompt.split('<ImageHere>')
        if len(prompt_segs) == 1:
            prompt_segs = [prompt, '']
        
        # Get context embedding
        context_emb = model.get_context_emb(prompt, img_list)
        
        # Prepare inputs for generation
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to('cuda') for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        t1 = time.time()
        
        # Generate with ONLY
        with torch.inference_mode():
            with torch.no_grad():
                # Use llama_model.generate() directly to support ONLY
                outputs = model.llama_model.generate(
                    inputs_embeds=context_emb,
                    attention_mask=torch.ones(context_emb.shape[:2], dtype=torch.long, device=context_emb.device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    use_only=args.use_only,
                    enhance_layer_index=args.enhance_layer_index,
                )
        
        t2 = time.time()
        print(f"Time: {t2-t1}")
        
        # Decode output
        input_token_len = context_emb.shape[1]
        output_ids = outputs[:, input_token_len:]
        outputs_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs_text = outputs_text.strip()
        
        # Remove stop words
        for stop_id in stop_words_ids:
            stop_text = tokenizer.decode(stop_id[0], skip_special_tokens=True)
            if outputs_text.endswith(stop_text):
                outputs_text = outputs_text[:-len(stop_text)]
        outputs_text = outputs_text.strip()

        print(f"[CHAIR Evaluation]")
        print(f"Image: {image_path}")
        print(f"Question: {qs}")
        print(f"Answer: {outputs_text}")
        print(f"="*50)

        img_save = {}
        img_save["image_id"] = img_id.item()
        img_save["caption"] = outputs_text

        # dump metric file
        with open(os.path.join(args.out_path, f"{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}_{args.max_new_tokens}_{args.method_name}.jsonl"), "a") as f:
            json.dump(img_save, f)
            f.write('\n')

        # write down time
        with open(os.path.join(args.out_path, f"{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}_{args.max_new_tokens}_{args.method_name}_time.txt"), "a") as f:
            f.write(f"{t2-t1}\n")
    
    if args.use_ritual:
        logger.info(f"RITUAL Transformation: {pos_aug_counter}")


if __name__ == "__main__":
    main()

