import os
import random
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from monotonic_align import mask_from_lens
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
from meldataset import get_dataloaders
from models import build_model, load_checkpoint, load_pretrained_models
from optimizers import build_optimizer
from utils import (
    configure_environment,
    get_image,
    length_to_mask,
    log_norm,
    maximum_path,
    recursive_munch,
)


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
def main(config_path):
    # Load config and set up environment
    config, logger, log_dir = configure_environment(config_path)

    # Initialize Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")
    device = accelerator.device

    # Read in configs
    batch_size = config.get("batch_size", 10)
    epochs = config.get("epochs_1st", 200)
    log_interval = config.get("log_interval", 10)
    max_len = config.get("max_len", 200)
    data_params = config.get("data_params", None)
    save_frequency = config.get("save_freq", 2)
    sr = config["preprocess_params"].get("sr", 24000)
    loss_params = Munch(config["loss_params"])
    TMA_epoch = loss_params.TMA_epoch

    # Load the datasets
    train_dataloader, val_dataloader, train_list = get_dataloaders(
        dataset_config=data_params, batch_size=batch_size, num_workers=4, device=device
    )

    # Load pretrained models
    with accelerator.main_process_first():
        text_aligner, pitch_extractor, plbert = load_pretrained_models(config)

    # Build model and optimizer
    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict={key: scheduler_params.copy() for key in model},
        lr=float(config["optimizer_params"].get("lr", 1e-4)),
    )

    # Prepare for accelerate training
    for k in model:
        model[k] = accelerator.prepare(model[k])
    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )
    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    # If resuming training, load checkpoint
    with accelerator.main_process_first():
        if config.get("pretrained_model", "") != "":
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                config["pretrained_model"],
                load_only_params=config.get("load_only_params", True),
            )
        else:
            start_epoch = 0
            iters = 0

    if hasattr(model.text_aligner, "module"):
        n_down = model.text_aligner.module.n_down
    else:
        n_down = model.text_aligner.n_down

    # Initialize losses
    stft_loss = MultiResolutionSTFTLoss().to(device)
    generator_loss = GeneratorLoss(model.mpd, model.msd).to(device)
    discriminator_loss = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wavlm_loss = WavLMLoss(
        model_params.slm.model, model.wd, sr, model_params.slm.sr
    ).to(device)

    # Train model
    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].train() for key in model]

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            waves, texts, input_lengths, mels, mel_input_length = process_batch(
                device, batch
            )

            with torch.no_grad():
                mel_mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")
                text_mask = length_to_mask(input_lengths).to(texts.device)

            """
            Use ASR text aligner to get
            1. ppgs (phoneme posteriorgrams): probability of each phoneme at each time step. used for style encoder?
            2. s2s_pred: seq2seq prediction: predicted phoneme at each time step. used for text encoder?
            3. s2s_attn: attention matrix: alignment between text and mel spectrogram. used for style encoder?
            """
            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mel_mask, texts)

            # Remove the first token from the attention matrix
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            # Mask the attention matrix
            with torch.no_grad():
                attn_mask = (
                    (~mel_mask)
                    .unsqueeze(-1)
                    .expand(mel_mask.shape[0], mel_mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mel_mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1
            s2s_attn.masked_fill_(attn_mask, 0.0)

            # encode the text
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            with torch.no_grad():
                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            # get clips
            mel_input_length_all = accelerator.gather(mel_input_length)
            shortest_mel_length = mel_input_length_all.min().item()
            mel_segment_len = min([int(shortest_mel_length / 2 - 1), max_len // 2])
            mel_segment_len_style = int(mel_input_length.min().item() / 2 - 1)

            en = []
            gt = []
            wav = []
            st = []

            for batch_index in range(len(mel_input_length)):
                mel_length_halved = int(mel_input_length[batch_index].item() / 2)

                # Extract segments for encoder and ground truth
                encoder_segment_start_index = np.random.randint(
                    0, mel_length_halved - mel_segment_len
                )
                en.append(
                    asr[
                        batch_index,
                        :,
                        encoder_segment_start_index : encoder_segment_start_index
                        + mel_segment_len,
                    ]
                )
                ## QUESTION: Why are we taking 2x the mel length?
                gt.append(
                    mels[
                        batch_index,
                        :,
                        (encoder_segment_start_index * 2) : (
                            (encoder_segment_start_index + mel_segment_len) * 2
                        ),
                    ]
                )

                # Extract corresponding waveform segments
                waveform_start_index = encoder_segment_start_index * 2 * 300
                waveform_end_index = (
                    (encoder_segment_start_index + mel_segment_len) * 2 * 300
                )
                waveform_segment = waves[batch_index][
                    waveform_start_index:waveform_end_index
                ]
                wav.append(torch.from_numpy(waveform_segment).to(device))

                # Extract style references (better to be different from the GT)
                style_segment_start_index = np.random.randint(
                    0, mel_length_halved - mel_segment_len_style
                )
                style_segment = mels[
                    batch_index,
                    :,
                    (style_segment_start_index * 2) : (
                        (style_segment_start_index + mel_segment_len_style) * 2
                    ),
                ]
                st.append(style_segment)

            en = torch.stack(en)  # text encoder output
            gt = torch.stack(gt).detach()  # ground truth mel spectrogram
            st = torch.stack(st).detach()  # style reference mel spectrogram
            wav = torch.stack(wav).float().detach()  # waveform

            # Check if the ground truth segment is too short for style encoding
            if gt.shape[-1] < 80:
                continue  # Skip this iteration if segment is too short

            # Prepare data for model input
            with torch.no_grad():
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                normalized_gt = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
            # Select appropriate input for style encoder based on whether the model is multispeaker
            style_input = st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            # Encode style
            style_encoding = model.style_encoder(style_input)
            # Decode synthesized speech from encoded text, pitch, normalized mel spectrograms, and style encoding
            ## Question: how is normalized_gt different used as energy?
            model_output = model.decoder(en, F0_real, normalized_gt, style_encoding)

            # Calculate discriminator loss
            d_loss = (
                discriminator_loss(
                    wav.detach().unsqueeze(1).float(), model_output.detach()
                ).mean()
                if epoch >= TMA_epoch
                else 0
            )

            # Calculate generator loss
            loss_mel = stft_loss(model_output.squeeze(), wav.detach())
            loss_s2s = loss_mono = loss_gen_all = loss_slm = 0
            if epoch >= TMA_epoch:
                for _s2s_pred, _text_input, _text_length in zip(
                    s2s_pred, texts, input_lengths
                ):
                    loss_s2s += F.cross_entropy(
                        _s2s_pred[:_text_length], _text_input[:_text_length]
                    )
                loss_s2s /= texts.size(0)
                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                loss_gen_all = generator_loss(
                    wav.detach().unsqueeze(1).float(), model_output
                ).mean()
                loss_slm = wavlm_loss(wav.detach(), model_output).mean()

                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * loss_gen_all
                    + loss_params.lambda_slm * loss_slm
                )
            else:
                g_loss = loss_mel

            # Accumulate running loss for logging
            running_loss += accelerator.gather(loss_mel).mean().item()

            # Backpropagate losses
            if epoch >= TMA_epoch:
                accelerator.backward(d_loss)
            accelerator.backward(g_loss)

            # Update model parameters
            optimizer.step("msd")
            optimizer.step("mpd")
            optimizer.step("text_encoder")
            optimizer.step("style_encoder")
            optimizer.step("decoder")
            if epoch >= TMA_epoch:
                optimizer.step("text_aligner")
                optimizer.step("pitch_extractor")

            iters += 1

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                logger.info(
                    "Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f"
                    % (
                        epoch + 1,
                        epochs,
                        i + 1,
                        len(train_list) // batch_size,
                        running_loss / log_interval,
                        loss_gen_all,
                        d_loss,
                        loss_mono,
                        loss_s2s,
                        loss_slm,
                    )
                )

                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/mono_loss", loss_mono, iters)
                writer.add_scalar("train/s2s_loss", loss_s2s, iters)
                writer.add_scalar("train/slm_loss", loss_slm, iters)

                running_loss = 0

                print("Time elasped:", time.time() - start_time)

        # Prepare for validation step
        loss_test = 0
        _ = [model[key].eval() for key in model]
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves, texts, input_lengths, mels, mel_input_length = process_batch(
                    device, batch
                )

                with torch.no_grad():
                    mel_mask = length_to_mask(mel_input_length // (2**n_down)).to(
                        "cuda"
                    )
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    """
                    Use ASR text aligner to get
                    1. ppgs (phoneme posteriorgrams): probability of each phoneme at each time step. used for style encoder?
                    2. s2s_pred: seq2seq prediction: predicted phoneme at each time step. used for text encoder?
                    3. s2s_attn: attention matrix: alignment between text and mel spectrogram. used for style encoder?
                    """
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mel_mask, texts)

                    # Remove the first token from the attention matrix
                    s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

                    # Mask the attention matrix
                    attn_mask = (
                        (~mel_mask)
                        .unsqueeze(-1)
                        .expand(
                            mel_mask.shape[0], mel_mask.shape[1], text_mask.shape[-1]
                        )
                        .float()
                        .transpose(-1, -2)
                    )
                    attn_mask = (
                        attn_mask.float()
                        * (~text_mask)
                        .unsqueeze(-1)
                        .expand(
                            text_mask.shape[0], text_mask.shape[1], mel_mask.shape[-1]
                        )
                        .float()
                    )
                    attn_mask = attn_mask < 1
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)

                asr = t_en @ s2s_attn

                # get clips
                mel_input_length_all = accelerator.gather(mel_input_length)
                mel_segment_len = min(
                    [int(mel_input_length.min().item() / 2 - 1), max_len // 2]
                )

                en = []
                gt = []
                wav = []

                for batch_index in range(len(mel_input_length)):
                    mel_length_halved = int(mel_input_length[batch_index].item() / 2)

                    # Extract segments for encoder and ground truth
                    encoder_segment_start_index = np.random.randint(
                        0, mel_length_halved - mel_segment_len
                    )
                    en.append(
                        asr[
                            batch_index,
                            :,
                            encoder_segment_start_index : encoder_segment_start_index
                            + mel_segment_len,
                        ]
                    )
                    gt.append(
                        mels[
                            batch_index,
                            :,
                            (encoder_segment_start_index * 2) : (
                                (encoder_segment_start_index + mel_segment_len) * 2
                            ),
                        ]
                    )

                    # Extract corresponding waveform segments
                    waveform_start_index = encoder_segment_start_index * 2 * 300
                    waveform_end_index = (
                        (encoder_segment_start_index + mel_segment_len) * 2 * 300
                    )
                    waveform_segment = waves[batch_index][
                        waveform_start_index:waveform_end_index
                    ]
                    wav.append(torch.from_numpy(waveform_segment).to(device))

                en = torch.stack(en)  # text encoder output
                gt = torch.stack(gt).detach()  # ground truth mel spectrogram
                wav = torch.stack(wav).float().detach()  # waveform

                # Prepare data for model input
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                normalized_gt = log_norm(gt.unsqueeze(1)).squeeze(1)
                style_encoding = model.style_encoder(gt.unsqueeze(1))

                # Decode synthesized speech from encoded text, pitch, normalized mel spectrograms, and style encoding
                model_output = model.decoder(en, F0_real, normalized_gt, style_encoding)

                # Calculate loss
                loss_mel = stft_loss(model_output.squeeze(), wav.detach())
                # Accumulate loss for logging
                loss_test += accelerator.gather(loss_mel).mean().item()

                iters_test += 1

        if accelerator.is_main_process:
            print("Epochs:", epoch + 1)
            logger.info("Validation loss: %.3f" % (loss_test / iters_test) + "\n\n\n\n")
            print("\n\n\n")
            writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch + 1)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure("eval/attn", attn_image, epoch)

            with torch.no_grad():
                for batch_index in range(len(asr)):
                    mel_length_halved = int(mel_input_length[batch_index].item())
                    gt = mels[batch_index, :, :mel_length_halved].unsqueeze(0)
                    en = asr[batch_index, :, : mel_length_halved // 2].unsqueeze(0)

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    F0_real = F0_real.unsqueeze(0)
                    style_encoding = model.style_encoder(gt.unsqueeze(1))
                    normalized_gt = log_norm(gt.unsqueeze(1)).squeeze(1)

                    model_output = model.decoder(
                        en, F0_real, normalized_gt, style_encoding
                    )

                    writer.add_audio(
                        "eval/y" + str(batch_index),
                        model_output.cpu().numpy().squeeze(),
                        epoch,
                        sample_rate=sr,
                    )
                    if epoch == 0:
                        writer.add_audio(
                            "gt/y" + str(batch_index),
                            waves[batch_index].squeeze(),
                            epoch,
                            sample_rate=sr,
                        )

                    if batch_index >= 6:
                        break

            if epoch % save_frequency == 0:
                if (loss_test / iters_test) < best_loss:
                    best_loss = loss_test / iters_test
                print("Saving..")
                state = {
                    "net": {key: model[key].state_dict() for key in model},
                    "optimizer": optimizer.state_dict(),
                    "iters": iters,
                    "val_loss": loss_test / iters_test,
                    "epoch": epoch,
                }
                save_path = os.path.join(log_dir, "epoch_1st_%05d.pth" % epoch)
                torch.save(state, save_path)


def process_batch(device, batch):
    waves = batch[0]
    batch = [b.to(device) for b in batch[1:]]
    texts, input_lengths, _, _, mels, mel_input_length, _ = batch
    return waves, texts, input_lengths, mels, mel_input_length


if __name__ == "__main__":
    main()
