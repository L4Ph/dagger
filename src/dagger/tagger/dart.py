from dartrs.dartrs import DartTokenizer
from dartrs.utils import get_generation_config
from dartrs.v2 import (
    compose_prompt,
    MixtralModel,
    V2Model,
)


def generate_dart_prompt(dart_prompt):
    DART_MODEL_NAME = "p1atdev/dart-v2-moe-sft"

    tokenizer = DartTokenizer.from_pretrained(DART_MODEL_NAME)
    model = MixtralModel.from_pretrained(DART_MODEL_NAME, dtype="fp16")

    config = get_generation_config(
        prompt=compose_prompt(
            copyright="mihoyo",
            character="seele_vollerei",
            rating="general",  # sfw, general, sensitive, nsfw, questionable, explicit
            aspect_ratio="tall",  # ultra_wide, wide, square, tall, ultra_tall
            length="medium",  # very_short, short, medium, long, very_long
            identity="none",  # none, lax, strict
            prompt=dart_prompt,
        ),
        tokenizer=tokenizer,
        top_p=1,
        top_k=250,
        temperature=1.0,
    )

    output = dart_prompt + model.generate(config)
    return output
