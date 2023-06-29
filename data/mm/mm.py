import os
import glob

import numpy as np
import torch
import pandas as pd


import decord
from decord import cpu
from decord import VideoReader

from youtube.get_gpt_descriptions import parse_output_subtitles
from gpt.src.tokenizer import GPTTokenizer

decord.bridge.set_bridge("torch")


class VideoFramesIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        data_folder: str,
        frame_transform=None,
        nframes_per_iteration: int = 29,
        nframes_per_video: int = 100,
        seed: int = 42,
        device: str = "cuda:0",
    ):
        super(VideoFramesIterator).__init__()

        self.annotations = annotations
        self.nframes_per_iteration = nframes_per_iteration
        self.nframes_per_video = nframes_per_video
        self.frame_transform = frame_transform
        self.data_folder = data_folder
        self.epoch_size = int(self.annotations.nbframes.sum() / nframes_per_iteration)
        self.seed = seed
        self.nframes_per_iteration = nframes_per_iteration
        self.rng = np.random.default_rng(seed)
        self.tokenizer = GPTTokenizer(model="gpt2", max_tokens=None)
        self.device = device

        # find mp4 files in data_folder
        file_ids = [
            os.path.basename(file) for file in glob.glob(data_folder + "/*.mp4")
        ]
        # filter annotations dataframe
        print("file ids", file_ids)
        print("before filtering: ", len(self.annotations))
        self.annotations = self.annotations[self.annotations.file.isin(file_ids)]
        print("after filtering: ", len(self.annotations))

    def __iter__(self):
        for _ in range(self.epoch_size):
            sample = self.annotations.sample(
                1, weights="nbframes", replace=False, random_state=self.rng
            )
            sample = sample.iloc[0]
            path = self.data_folder + sample["file"]
            nbframes = float(sample["nbframes"])
            start = self.rng.uniform(0, nbframes - self.nframes_per_iteration)
            length = self.rng.uniform(30, 2000)
            end = min(start + length, nbframes)

            subtitle = path.replace("youtube/dgx_videos/", "samplesv4").replace(
                ".mp4", ".gpt35"
            )
            parsed_subtitle = parse_output_subtitles(subtitle)

            video_annotations = []
            video_frames = []
            if not os.path.exists(subtitle):
                print("subtitle does not exist: ", subtitle)
                continue
            else:
                with open(subtitle, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
                    for t in range(len(parsed_subtitle[0])):
                        sub_start, sub_stop = parsed_subtitle[0][t]
                        lines = parsed_subtitle[1][t]

                        if sub_start < end and sub_stop > start:
                            annotation = (
                                "<|startoftext|> "
                                + "\n".join(lines)
                                + "<"
                                + str(sub_start)
                                + ", "
                                + str(sub_stop)
                                + ">"
                                + " <|endoftext|>"
                            )

                            annotation = torch.tensor(
                                self.tokenizer.encode(
                                    annotation, allowed_special="all"
                                ),
                                dtype=torch.long,
                            ).contiguous()

                            video_annotations.append(annotation)

                            sub_length = sub_stop - sub_start
                            if sub_length > self.nframes_per_video:
                                indices = np.linspace(
                                    sub_start,
                                    sub_stop,
                                    self.nframes_per_video,
                                    endpoint=True,
                                )
                                # round indices
                                indices = np.round(indices).astype(int)
                            else:
                                indices = np.arange(start, end)

                            vr = VideoReader(path, width=512, height=256, ctx=cpu(0))
                            frames = vr.get_batch(indices)
                            # permute from (T, H, W, C) to (T, C, H, W)
                            frames = frames.permute(0, 3, 1, 2)
                            frames = 2 * (frames / 255.0) - 1.0
                            video_frames.append(frames)

            if len(video_frames) != 0 and len(video_annotations) != 0:
                output = {"frames": video_frames, "annotations": video_annotations}
                yield output

    def collate_fn(self, batch):
        return batch


if __name__ == "__main__":
    import argparse

    root = "/afs/csail.mit.edu/u/a/akyurek/align/git/MineCLIP"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=f"{root}/youtube/samplesv4/")
    parser.add_argument(
        "--annotations", type=str, default=f"{root}/youtube/dgx_videos.csv"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--nframes_per_iteration", type=int, default=29)
    parser.add_argument("--nframes_per_video", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = VideoFramesIterator(
        annotations=annotations,
        data_folder=args.data_folder,
        nframes_per_iteration=args.nframes_per_iteration,
        nframes_per_video=args.nframes_per_video,
        seed=args.seed,
        device=device,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    from mmgpt.nanoGPT.model import GPTConfig, GPT
    from youtube.vae_finetune import AutoencoderKL, AutoencoderKLWLoss

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="vae",
    )

    vae = AutoencoderKLWLoss(vae, kl_weight=1.0, pixelloss_weight=1.0).to(device)

    vae.load_state_dict(torch.load(f"{root}/youtube/artifacts/vae_state_dict.pt", map_location=device))
    # remove vae decoder
    vae.vae.decoder = None
    # disable vae gradients
    for param in vae.parameters():
        param.requires_grad = False
        param.grad = None

    config = GPTConfig(n_embd=2048, n_layer=8, n_head=2, block_size=2048)
    config.vae_size = 8192
    model = GPT(config).to(device)

    for batch in dataloader:
        batched_frames = []
        batched_annotations = []
        for index, instance in enumerate(batch):
            frames = instance["frames"]
            annotations = instance["annotations"]

            for t, annotation in enumerate(annotations):
                try:
                    annotations[t] = annotation.cuda()
                except RuntimeError:
                    breakpoint()

            batched_annotations.append(annotations)

            for t, frame in enumerate(frames):
                # chunk batches to smaller batches
                features = []

                for v in range(0, len(frame), 10):
                    feat = vae.encode(frame[v : v + 10].to(device)).mean
                    feat = torch.flatten(feat, start_dim=1).contiguous()
                    features.append(feat)

                features = torch.cat(features, dim=0)
                frames[t] = features
            batched_frames.append(frames)

        loss = model.forward(batched_annotations, batched_frames)

