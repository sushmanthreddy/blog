+++
title =  "GSOC WEEK 1"
tags = ["DevoLearn", "GSOC","OpenWorm" ,"INCF"]
date = "2023-06-10"

+++

Gsoc coding period week 1 have started coding and setup is complete the sam model is heavy and can,t be trained on local computer ,so most of my work will  be using kaggle gpu.
Here is little explanation about [sam(segment anything model)](https://github.com/facebookresearch/segment-anything)

Here we have three weights for the model VIT-H and VIT-l model where the large model is of 636m parameter and 350m parameters .so fine tuning such large model will be a big mistake ,It will take aproximately three days to train the whole model.
So we take small model VIT-b model of 90 million parameters is also takes lot of time ,but here we train only the part of model than complete model.

# Model architecture 

```bash
  Sam(
  (image_encoder): ImageEncoderViT(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (blocks): ModuleList(
      (0-11): 12 x Block(
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=768, out_features=2304, bias=True)
          (proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (lin1): Linear(in_features=768, out_features=3072, bias=True)
          (lin2): Linear(in_features=3072, out_features=768, bias=True)
          (act): GELU(approximate='none')
        )
      )
    )
    (neck): Sequential(
      (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): LayerNorm2d()
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (3): LayerNorm2d()
    )
  )
  (prompt_encoder): PromptEncoder(
    (pe_layer): PositionEmbeddingRandom()
    (point_embeddings): ModuleList(
      (0-3): 4 x Embedding(1, 256)
    )
    (not_a_point_embed): Embedding(1, 256)
    (mask_downscaling): Sequential(
      (0): Conv2d(1, 4, kernel_size=(2, 2), stride=(2, 2))
      (1): LayerNorm2d()
      (2): GELU(approximate='none')
      (3): Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2))
      (4): LayerNorm2d()
      (5): GELU(approximate='none')
      (6): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (no_mask_embed): Embedding(1, 256)
  )
  (mask_decoder): MaskDecoder(
    (transformer): TwoWayTransformer(
      (layers): ModuleList(
        (0-1): 2 x TwoWayAttentionBlock(
          (self_attn): Attention(
            (q_proj): Linear(in_features=256, out_features=256, bias=True)
            (k_proj): Linear(in_features=256, out_features=256, bias=True)
            (v_proj): Linear(in_features=256, out_features=256, bias=True)
            (out_proj): Linear(in_features=256, out_features=256, bias=True)
          )
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (cross_attn_token_to_image): Attention(
            (q_proj): Linear(in_features=256, out_features=128, bias=True)
            (k_proj): Linear(in_features=256, out_features=128, bias=True)
            (v_proj): Linear(in_features=256, out_features=128, bias=True)
            (out_proj): Linear(in_features=128, out_features=256, bias=True)
          )
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (mlp): MLPBlock(
            (lin1): Linear(in_features=256, out_features=2048, bias=True)
            (lin2): Linear(in_features=2048, out_features=256, bias=True)
            (act): ReLU()
          )
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (cross_attn_image_to_token): Attention(
            (q_proj): Linear(in_features=256, out_features=128, bias=True)
            (k_proj): Linear(in_features=256, out_features=128, bias=True)
            (v_proj): Linear(in_features=256, out_features=128, bias=True)
            (out_proj): Linear(in_features=128, out_features=256, bias=True)
          )
        )
      )
      (final_attn_token_to_image): Attention(
        (q_proj): Linear(in_features=256, out_features=128, bias=True)
        (k_proj): Linear(in_features=256, out_features=128, bias=True)
        (v_proj): Linear(in_features=256, out_features=128, bias=True)
        (out_proj): Linear(in_features=128, out_features=256, bias=True)
      )
      (norm_final_attn): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (iou_token): Embedding(1, 256)
    (mask_tokens): Embedding(4, 256)
    (output_upscaling): Sequential(
      (0): ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2))
      (1): LayerNorm2d()
      (2): GELU(approximate='none')
      (3): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
      (4): GELU(approximate='none')
    )
    (output_hypernetworks_mlps): ModuleList(
      (0-3): 4 x MLP(
        (layers): ModuleList(
          (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
          (2): Linear(in_features=256, out_features=32, bias=True)
        )
      )
    )
    (iou_prediction_head): MLP(
      (layers): ModuleList(
        (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
        (2): Linear(in_features=256, out_features=4, bias=True)
      )
    )
  )
)

```


# SAM(segment anything)

After reading the complete model here is what my understanding !
The model is divided into three categories:
1. ImageEncoderViT
2. Prompt_encoder
3. Mask Decoder 

## ImageEncoderViT

Here the architecture of the image encoder is Vision transformer. Here the model expects a colored image and whole sam model is trained on color images of phone data , and image encoder omits the shape of 
When the model expects the image shape of 

```bash
(3,1024,10244)
```
here the 3 represnts the red ,green and blue channels and 1024 represents the height and width. When the image is feed through the image encoder the image encoder omits the shape of 

```bash
(B,256,64,64)
```
B represents the batch size. Here is the image encoder structure.

```bash
(image_encoder): ImageEncoderViT(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (blocks): ModuleList(
      (0-11): 12 x Block(
        (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=768, out_features=2304, bias=True)
          (proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (lin1): Linear(in_features=768, out_features=3072, bias=True)
          (lin2): Linear(in_features=3072, out_features=768, bias=True)
          (act): GELU(approximate='none')
        )
      )
    )
    (neck): Sequential(
      (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): LayerNorm2d()
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (3): LayerNorm2d()
    )
  )
  ```
  ## Prompt Encoder

  Here in sam model has another block called prompt encoder ,where it takes inputs as masks, points, bounding box. Where these prompts are feed through prompt encoder and prompt embedding are feed for the mask decoder.
  The prompt shape will be 

  ```bash
  B,number_of_boxes,4
  ```
  here the B reprsent the batch size. prompt encoder shape is there below.

  ```bash
  (prompt_encoder): PromptEncoder(
    (pe_layer): PositionEmbeddingRandom()
    (point_embeddings): ModuleList(
      (0-3): 4 x Embedding(1, 256)
    )
    (not_a_point_embed): Embedding(1, 256)
    (mask_downscaling): Sequential(
      (0): Conv2d(1, 4, kernel_size=(2, 2), stride=(2, 2))
      (1): LayerNorm2d()
      (2): GELU(approximate='none')
      (3): Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2))
      (4): LayerNorm2d()
      (5): GELU(approximate='none')
      (6): Conv2d(16, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (no_mask_embed): Embedding(1, 256)
  )
  ```

  ## Mask decoder
  Mask decoder takes the input from the image encoder and prompt encoder and produces the masks.
  As main proble with our data is that they are grey scale and sam model is trained on colored images 
  so it will not work on the grey scale images .
  So we need to convert the images into the colored images rather than greyscale.
  so most of work went on understanding the sam model.In week 1 of gsoc most of the time went on model architecture.
  
  Here is the correct explanation of the video.

  <img src="../images/gsoc_week1/section-3.1c.gif" alt="" width="600" height="">

  And most the of week understood and converted into colored images







