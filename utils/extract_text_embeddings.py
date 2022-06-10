# copied from https://github.com/chongzhou96/DenseCLIP/blob/master/tools/denseclip_utils/prompt_engineering.py
from typing import Dict, List, Optional
import pickle as pkl
import torch
import clip
from tqdm import tqdm


prompt_templates: List[str] = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    'there is a {} in the scene.',
    'there is the {} in the scene.',
    'this is a {} in the scene.',
    'this is the {} in the scene.',
    'this is one {} in the scene.',
]


@torch.no_grad()
def extract_text_embeddings(
        model,
        categories: List[str],
        templates: List[str]
) -> Dict[str, torch.FloatTensor]:
    cat_to_text_embedding: Dict[str, torch.FloatTensor] = dict()
    for category in tqdm(categories):
        texts = [template.format(category) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        text_embeddings = model.encode_text(texts)  # embed with text encoder
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        avg_text_embedding = text_embeddings.mean(dim=0)  # text_dim
        avg_text_embedding /= avg_text_embedding.norm()
        cat_to_text_embedding[category] = avg_text_embedding.to(torch.float32)  # float16 -> float32
    return cat_to_text_embedding


def prompt_engineering(
        categories: List[str],
        model_name: str = "RN50",
        fp: Optional[str] = None,
        device: torch.device = torch.device("cuda:0")
) -> Dict[str, torch.FloatTensor]:
    if isinstance(model_name, str):
        assert model_name in ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],\
            ValueError(model_name)
        print(f"extracting text embeddings using {model_name}...")
        model, preprocess = clip.load(model_name, device=device)
    else:
        model = model_name

    cat_to_text_embedding: Dict[str, torch.FloatTensor] = extract_text_embeddings(
        model=model, categories=categories, templates=prompt_templates
    )

    if fp is not None:
        pkl.dump(cat_to_text_embedding, open(fp, "wb"))
        print(f"cat_to_text_embedding file saved at {fp}")
    return cat_to_text_embedding
