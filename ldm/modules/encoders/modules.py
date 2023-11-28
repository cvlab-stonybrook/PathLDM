import torch
import torch.nn as nn
from functools import partial
import math
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
import open_clip
from ldm.modules.x_transformer import (
    Encoder,
    TransformerWrapper,
)  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device="cuda",
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


def clip_transformer_forward(model, input_ids_list, attention_mask, class_embed=None):
    # this is a hack to get the CLIP transformer to work with long captions
    # class_embed is concatenated to the input embeddings

    output_attentions = model.config.output_attentions
    output_hidden_states = model.config.output_hidden_states
    return_dict = model.config.use_return_dict

    sz = input_ids_list[0].size()
    input_shape = (sz[0], sz[1] * len(input_ids_list))

    hidden_states_list = []

    for input_ids in input_ids_list:
        hidden_states = model.embeddings(input_ids)
        hidden_states_list.append(hidden_states)

    hidden_states = torch.cat(hidden_states_list, dim=1)

    if class_embed is not None:
        input_shape = (input_shape[0], 1 + input_shape[1])
        class_embed = class_embed.unsqueeze(1)
        hidden_states = torch.cat([class_embed, hidden_states], dim=1)

    # causal mask is applied over the whole sequence (154 tokens)
    causal_attention_mask = _make_causal_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )

    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = model.final_layer_norm(last_hidden_state)

    return last_hidden_state


class FrozenCLIPEmbedder(nn.Module):
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.clip_max_length = self.tokenizer.model_max_length
        except:
            # when using plip model
            self.tokenizer = AutoTokenizer.from_pretrained(version)
            self.clip_max_length = 77
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = self.clip_max_length * math.ceil(
            max_length / self.clip_max_length
        )
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)

        if input_ids.shape[1] != self.clip_max_length:
            input_ids_list = input_ids.split(self.clip_max_length, dim=-1)
        else:
            input_ids_list = [input_ids]

        z = clip_transformer_forward(self.transformer.text_model, input_ids_list, attention_mask)
        return z

    @torch.no_grad()
    def encode(self, text):
        return self(text)


class BioMedCLIPEmbedder(nn.Module):
    """Uses microsoft Biomed CLIP transformer (from hf, based on openclip)
    has a max context length of 256
    """

    def __init__(self, device="cuda", max_length=77):
        super().__init__()
        version = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.clip, _, _ = open_clip.create_model_and_transforms(
            f"hf-hub:{version}"
        )
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{version}")

        self.max_length = max_length
        self.device = device
        self.freeze()

    def freeze(self):
        self.clip = self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, text):
        tokens = self.tokenizer(text, context_length=self.max_length).to(self.device)

        z = self.clip.text.transformer(tokens)[0]
        return z

    def forward(self, text):
        return self.encode(text)
