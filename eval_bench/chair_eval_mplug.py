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

# Add Nullu to path for mplug_owl2
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nullu_path = os.path.abspath(os.path.join(base_dir, 'Nullu'))
if nullu_path not in sys.path:
    sys.path.insert(0, nullu_path)

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

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
    parser = argparse.ArgumentParser(description="CHAIR evaluation on mPLUG-Owl2 with ONLY.")
    parser.add_argument("--model_path", type=str, help="path to mPLUG-Owl2 model")
    parser.add_argument("--model_base", type=str, default="mplug")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="./logs/chair")
    parser.add_argument("--out_path", type=str, default="./chair_results/mplug", help="output path")

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
        model_string_name = "mplug_owl2"
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing mPLUG-Owl2 Model')

    # Load mPLUG-Owl2 model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit=False, load_4bit=False, device=device
    )

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=image_processor,
        model="mplug"
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
        'crop': v2.RandomResizedCrop(size=448),
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
            raw_image = Image.open(image_path).convert('RGB')
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                max_edge = max(raw_image_pos.size)
                raw_image_pos = raw_image_pos.resize((max_edge, max_edge))
                image_pos = process_images([raw_image_pos], image_processor)
                image_pos = image_pos.to(device).half()
            pos_aug_counter[pos_aug] += 1
            logger.info(f"RITUAL Transformation: {pos_aug}")
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)

        # Prepare image
        raw_image = Image.open(image_path).convert('RGB')
        max_edge = max(raw_image.size)
        raw_image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([raw_image], image_processor)
        image_tensor = image_tensor.to(device).half()
        
        # Prepare prompt
        conv = conv_templates["mplug_owl2"].copy()
        prompt = DEFAULT_IMAGE_TOKEN + qs
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        t1 = time.time()
        
        # Generate with ONLY
        with torch.inference_mode():
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    use_only=args.use_only,
                    enhance_layer_index=args.enhance_layer_index,
                )
        
        t2 = time.time()
        print(f"Time: {t2-t1}")
        
        # Decode output
        input_token_len = input_ids.shape[1]
        output_ids = outputs[:, input_token_len:]
        outputs_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Remove stop words
        if outputs_text.endswith(stop_str):
            outputs_text = outputs_text[:-len(stop_str)]
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

