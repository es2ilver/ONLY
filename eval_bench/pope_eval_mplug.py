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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Nullu'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Nullu', 'mplug_owl2'))

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import dist_util
from utils.logger import create_logger
from pope_loader import POPEDataSet

from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings(action='ignore')

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
    parser = argparse.ArgumentParser(description="POPE evaluation on mPLUG-Owl2.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/mplug-owl2-llama2-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="./logs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)
    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.2)

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--type", type=str, default="random")
    parser.add_argument("--dataset_name", type=str, default="coco")

    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / float(TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    
    return pred_list

def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()
    
    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        if args.use_ritual:
            method_name = "RITUAL"
        elif args.use_vcd:
            method_name = "VCD"
        elif args.use_m3id:
            method_name = "M3ID"
        elif args.use_only:
            method_name = "ONLY"
        else:
            method_name = "Regular"
        experiment_dir = f"{args.log_path}/pope/{model_string_name}/{method_name}_{args.dataset_name}_{args.type}_{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}_layer_{args.enhance_layer_index}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing mPLUG-Owl2 Model')

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit=False, load_4bit=False, device=device
    )

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model="mplug"
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

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
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"]
        label_list = label_list + list(label)

        image_pos = None
        image_neg = None

        if args.use_ritual:
            raw_image = Image.open(image_path[0]).convert('RGB')
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                max_edge = max(raw_image_pos.size)
                raw_image_pos = raw_image_pos.resize((max_edge, max_edge))
                image_pos = process_images([raw_image_pos], image_processor)
                image_pos = image_pos.to(device).half()
            pos_aug_counter[pos_aug] += 1
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)

        # Prepare image
        raw_image = Image.open(image_path[0]).convert('RGB')
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
        
        # Decode output
        input_token_len = input_ids.shape[1]
        output_ids = outputs[:, input_token_len:]
        outputs_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Remove stop words
        if outputs_text.endswith(stop_str):
            outputs_text = outputs_text[:-len(stop_str)]
        outputs_text = outputs_text.strip()
        
        pred_list = recorder(outputs_text, pred_list)
        print(f"[VQA for POPE]")
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        print(f"A: {outputs_text}")

        if label == 1: print(f"GT: Yes")
        elif label == 0: print(f"GT: No")

        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        print(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        
        print(f"="*50)

    if len(pred_list) != 0:
        logger.info(vars(args))
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        if args.use_ritual:
            logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()

