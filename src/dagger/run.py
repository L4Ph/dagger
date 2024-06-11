import sys
from typing import Generator, Iterable
from tagger.interrogator import Interrogator
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
from dartrs.dartrs import DartTokenizer
from dartrs.utils import get_generation_config
from dartrs.v2 import (
    compose_prompt,
    MixtralModel,
    V2Model,
)

from tagger.interrogators import interrogators

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--dir", help="Predictions for all images in the directory")
group.add_argument("--file", help="Predictions for one file")

parser.add_argument(
    "--threshold",
    type=float,
    default=0.35,
    help="Prediction threshold (default is 0.35)",
)
parser.add_argument(
    "--ext",
    default=".txt",
    help="Extension to add to caption file in case of dir option (default is .txt)",
)
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite caption file if it exists"
)
parser.add_argument("--cpu", action="store_true", help="Use CPU only")
parser.add_argument(
    "--rawtag", action="store_true", help="Use the raw output of the model"
)
parser.add_argument(
    "--recursive", action="store_true", help="Enable recursive file search"
)
parser.add_argument(
    "--exclude-tag",
    dest="exclude_tags",
    action="append",
    metavar="t1,t2,t3",
    help="Specify tags to exclude (Need comma-separated list)",
)
parser.add_argument(
    "--model",
    default="wd14-convnextv2.v1",
    choices=list(interrogators.keys()),
    help="modelname to use for prediction (default is wd14-convnextv2.v1)",
)
args = parser.parse_args()

# get interrogator configs
interrogator = interrogators[args.model]

if args.cpu:
    interrogator.use_cpu()


def parse_exclude_tags() -> set[str]:
    if args.exclude_tags is None:
        return set()

    tags = []
    for str in args.exclude_tags:
        for tag in str.split(","):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(" ", "_").replace("\(", "(").replace("\)", ")")
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates


def image_interrogate(
    image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]
) -> list[tuple[str, float]]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    filtered_tags = [(tag, conf) for tag, conf in result[1] if conf >= args.threshold]
    print(f"Number of tags after filtering: {len(filtered_tags)}")

    postprocessed_tags = Interrogator.postprocess_tags(
        filtered_tags,
        threshold=args.threshold,
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags,
    )

    return postprocessed_tags


def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
    """
    Explore files by folder path
    """
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            yield path
        elif args.recursive and path.is_dir():
            yield from explore_image_files(path)


if args.dir:
    root_path = Path(args.dir)
    for image_path in explore_image_files(root_path):
        caption_path = image_path.parent / f"{image_path.stem}{args.ext}"

        if caption_path.is_file() and not args.overwrite:
            # skip if caption exists
            print("skip:", image_path)
            continue

        print("processing:", image_path)
        tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags())

        for tag, confidence in tags:
            print(f"{tag} : {confidence:.3f}")

        tags_str = ", ".join([tag for tag, _ in tags])

        with open(caption_path, "w") as fp:
            fp.write(tags_str)


if args.file:
    image_path = Path(args.file)
    tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags())

    # Filter tags with confidence >= 0.85
    high_confidence_tags = [tag for tag, confidence in tags if confidence >= 0.8]

    # # Create a filename with the current date and time
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # txt_filename = Path(f"{current_time}.txt")

    # print(f"High confidence tags have been written to {txt_filename}")

    # if len(sys.argv) != 2:
    #     print("Usage: python dart.py {tags}")
    #     sys.exit(1)

    dart_prompt = ", ".join(high_confidence_tags)
    # print(f"Tags with confidence >= 0.85: {dart_prompt}")

    # DART_MODEL_NAME = "p1atdev/dart-v2-moe-sft"

    # model = MixtralModel.from_pretrained(DART_MODEL_NAME)
    # tokenizer = DartTokenizer.from_pretrained(DART_MODEL_NAME)

    # config = get_generation_config(
    #     prompt=compose_prompt(
    #         copyright="mihoyo",
    #         character="seele_vollerei",
    #         rating="general",  # sfw, general, sensitive, nsfw, questionable, explicit
    #         aspect_ratio="tall",  # ultra_wide, wide, square, tall, ultra_tall
    #         length="long",  # very_short, short, medium, long, very_long
    #         identity="none",  # none, lax, strict
    #         prompt=dart_prompt,
    #     ),
    #     tokenizer=tokenizer,
    #     temperature=1.0,
    #     top_p=1,
    #     top_k=100,
    # )

    # output = model.generate(config)
    # print(f"Dart output: {output}")

    #####

    DART_MODEL_NAME = "p1atdev/dart-v2-moe-sft"

    tokenizer = AutoTokenizer.from_pretrained(DART_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        DART_MODEL_NAME, torch_dtype=torch.bfloat16
    )

    prompt = (
        f"<|bos|>"
        f"<copyright>mihoyo</copyright>"
        f"<character>seele_vollerei</character>"
        f"<|rating:general|><|aspect_ratio:tall|><|length:long|>"
        f"<general>{dart_prompt}<|identity:none|><|input_end|>"
    )
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=100,
            max_new_tokens=250,
            num_beams=1,
        )

    print(
        ", ".join(
            [
                tag
                for tag in tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
                if tag.strip() != ""
            ]
        )
    )
