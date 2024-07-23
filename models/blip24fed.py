import torch
import torch.nn.functional as F
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

class Blip24Fed(Blip2Qformer):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    def forward(self, samples, match_head="itm"):
        image = samples["image"]
        caption = samples["text_input"]

        image_feat = None
        text_feat = None

        if image:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feat = F.normalize(
                self.vision_proj(query_output.last_hidden_state).mean(dim=1, keepdim=False), dim=-1
            )

        if caption:
            text = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(caption.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

        return image_feat, text_feat