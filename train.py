import time
import argparse
import json
import logging
import math
import os
# from tqdm import tqdm
from pathlib import Path

import random
import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import soundfile as sf
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, AudioDiffusion
from transformers import SchedulerType, get_scheduler
from transformers.optimization import Adafactor, AdafactorSchedule

import logging
import pdb


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default="/data4/xiongchenxu/tango/tango/data/train_toy_fixed.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="/data4/xiongchenxu/tango/tango/data/valid_toy_fixed.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default= None,
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--reference_column",type=str,default="ref_loc",
        help= "The name of the column in the datasets containing the reference audios"
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--augment", action="store_true", default=False,
        help="Augment training data.",
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    # parser.add_argument(
    #     "--learning_rate", type=float, default=3e-4,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--prior_regulization_rate", type=int, default=0.5,
        help="For instance, if the num_train_epochs is 40, prior regulization will be activated in the first rate*epochs, which is 20",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value
    return new_state_dict


class Text2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_column, reference_column,num_examples=-1):
        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.reference = list(dataset[reference_column])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, text in zip(self.indices, self.audios, inputs):
            self.mapper[index] = [audio, text]

        if num_examples != -1:
            self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4 = self.inputs[index], self.audios[index], self.indices[index], self.reference[index]
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    # if args.augment and args.per_device_train_batch_size < 2:
    #     raise ValueError("Augmentation requires per device train batch size of at least 2.")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project="Text to Audio Diffusion")

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        if args.validation_file is not None:
            data_files["test"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column,reference_column = args.text_column, args.audio_column,args.reference_column

    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

    model = AudioDiffusion(
        args.text_encoder_name, args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.freeze_text_encoder, args.uncondition
    )
    
    if args.hf_model:
        hf_model_path = snapshot_download(repo_id=args.hf_model)
        model.load_state_dict(torch.load("{}/pytorch_model_main.bin".format(hf_model_path), map_location="cpu"))
        accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
        
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""

    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(raw_datasets["train"], prefix, text_column, audio_column, reference_column,args.num_examples)
        eval_dataset = Text2AudioDataset(raw_datasets["validation"], prefix, text_column, audio_column, reference_column,args.num_examples)
        test_dataset = Text2AudioDataset(raw_datasets["test"], prefix, text_column, audio_column, reference_column,args.num_examples)

        accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances(), test_dataset.get_num_instances()))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=test_dataset.collate_fn)

    # Optimizer
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
        if args.unet_model_config:
            optimizer_parameters = model.unet.parameters()
            accelerator.print("Optimizing UNet parameters.")
        else:
            optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
            accelerator.print("Optimizing UNet and channel transformer parameters.")
    else:
        optimizer_parameters = model.parameters()
        accelerator.print("Optimizing Text Encoder and UNet parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = Adafactor(optimizer_parameters, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = AdafactorSchedule(optimizer)

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            # 加载checkpoint的state_dict
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
            # 加载到模型中并检查对上和没对上的键
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            # 打印结果
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
            accelerator.print(f"Number of matched keys: {len(checkpoint) - len(missing_keys)}")
            accelerator.print(f"Number of missing keys: {len(missing_keys)}")
            accelerator.print(f"Number of unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                accelerator.print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                accelerator.print(f"Unexpected keys: {unexpected_keys}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf

    prior_reg = True
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                device = model.device
                text, audios, _ , reference= batch
                reference = [random.choice(ref) for ref in reference]
                target_length = int(duration * 102.4)

                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                    mel = mel.unsqueeze(1).to(device)
                    
                    # if args.augment and len(text) > 1:
                    #     # the last batch of the training data may have only one instance
                    #     # we check the length here so that the augmentation function doesn't throw an error
                    #     mixed_mel, _, _, mixed_captions = torch_tools.augment_wav_to_fbank(audios, text, len(audios), target_length, stft)
                    #     mixed_mel = mixed_mel.unsqueeze(1).to(device)
                    #     mel = torch.cat([mel, mixed_mel], 0)
                    #     text += mixed_captions
                    
                    true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                # Log inputs, outputs, and loss
                if epoch >args.prior_regulization_rate*args.num_train_epochs:
                    prior_reg = False
                
                loss = model(true_latent, text, reference, prior_reg, validation_mode=False)
                
                
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        model.uncondition = False

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                device = model.device
                text, audios, _,reference = batch
                reference = [random.choice(ref) for ref in reference]
                target_length = int(duration * 102.4)

                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                mel = mel.unsqueeze(1).to(device)
                true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                val_loss = model(true_latent, text, reference, prior_reg, validation_mode=True)
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch + 1
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item() / len(train_dataloader), 4)
            if completed_steps <= 5:
                print(f"loss: {total_loss.item()}")
                print(f'num: {len(train_dataloader)}')
            result["val_loss"] = round(total_val_loss.item() / len(eval_dataloader), 4)

            wandb.log(result)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(epoch, result["train_loss"], result["val_loss"])
            accelerator.print(result_string)

            with open(f"{args.output_dir}/summary.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                torch.save(model.state_dict(), f"{args.output_dir}/best_model.bin")

            if (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), f"{args.output_dir}/epoch_{epoch + 1}_model.bin")

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            torch.save(model.state_dict(), f"{args.output_dir}/epoch_{epoch + 1}_model.bin")
            
            
if __name__ == "__main__":
    main()
