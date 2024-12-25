# Video-LLM

### 🔥🔥🔥[Do Language Models Understand Time?](https://arxiv.org/abs/2412.13845)🤔

![image](./images/cover.png)
Do language models understand time?🧐 In the kitchen arena🧑‍🍳, where burritos are rolled🌯, rice waits patiently🍚, and sauce steals the spotlight, LLMs try their best to keep up. Captions flow like a recipe—precise and tempting—but can they truly tell the difference between prepping, cooking, and eating? After all, in cooking, timing isn’t just everything—it’s the secret sauce!🥳🥳🥳
>
> A collection of papers and resources related to Large Language Models in video domain. 
>
> More details please refer to our [paper](https://arxiv.org/abs/2412.13845). 
>
> Please let us know if you find out a mistake or have any suggestions by e-mail: Xi.Ding1@anu.edu.au
>
> If you find our paper useful for your research, please cite the following:

```
@article{ding2024language,
  title={Do Language Models Understand Time?},
  author={Ding, Xi and Wang, Lei},
  journal={arXiv preprint arXiv:2412.13845},
  year={2024}
}
```

---

## Models with Pretrained Image Encoder

### ResNet

<details>
<summary>Click to expand Table 1</summary>

| Model          | Venue        | Other modality encoders               | Interaction / Fusion mechanism                    | Description                                     |
|----------------|--------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------|
| [Flamingo](https://arxiv.org/abs/2204.14198)       | NeurIPS 2022 | Text: Chinchilla                     | Perceiver Resampler & Gated XATTN-DENSE          | Visual-language model for few-shot learning.   |

</details>

### CLIP ViT

<details>
<summary>Click to expand Table 2</summary>

| Model          | Venue        | Other modality encoders               | Interaction / Fusion mechanism                    | Description                                     |
|----------------|--------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------|
| mPLUG-2        | ICML 2023    | Text: BERT                           | Universal layers & cross-attention modules       | Modularized multi-modal foundation model.      |

| Vid2Seq        | CVPR 2023    | Text: T5-Base                        | Cross-modal attention                             | Sequence-to-sequence video-language model.     |
| Video-LLaMA    | EMNLP 2023   | Text: Vicuna, Audio: ImageBind       | Aligned via Q-Formers for video and audio         | Instruction-tuned multimodal model.           |
| ChatVideo      | arXiv 2023   | Text: ChatGPT, Audio: Whisper        | Tracklet-centric with ChatGPT reasoning           | Chat-based video understanding system.         |
| Video-ChatGPT  | ACL 2023     | Text: Vicuna-v1.1                    | Spatiotemporal features projected via linear layer| Integration of vision and language for video understanding. |
| Valley         | arXiv 2023   | Text: StableVicuna                   | Projection layer                                  | LLM for video assistant tasks.                |
| Macaw-LLM      | arXiv 2023   | Text: LLAMA-7B, Audio: Whisper       | Alignment module unifies multi-modal representations| Multimodal integration using image, audio, and video inputs. |
| Auto-AD II     | CVPR 2023    | Text: BERT                           | Cross-attention layers                            | Movie description using vision and language.   |
| GPT4Video      | ACMMM 2023   | Text: LLaMA 2                        | Transformer-based cross-attention layer           | Video understanding with LLM-based reasoning.  |
| LLaMA-VID      | ECCV 2023    | Text: Vicuna                         | Context attention and linear projector            | LLaMA-VID for visual-textual alignment in video. |
| COSMO          | arXiv 2024   | Text: OPT-IML/RedPajama/Mistral      | Gated cross-attention                              | Contrastive-streamlined multimodal model.      |
| VTimeLLM       | CVPR 2024    | Text: Vicuna                         | Linear layer                                      | Temporal video understanding enhanced with LLMs. |
| VILA           | CVPR 2024    | Text: LLaMA-2-7B/13B                 | Linear layer                                      | Vision-language model.                         |
| PLLaVA         | arXiv 2024   | Text: LLAMA-7B                       | MM projector with adaptive pooling                | Parameter-free extension for video captioning tasks. |
| V2Xum-LLaMA    | arXiv 2024   | Text: LLaMA 2                        | Vision adapter                                    | Video summarization using temporal prompt tuning. |
| VideoChat2     | CVPR 2024    | Text: Vicuna                         | Linear projection                                 | A comprehensive multi-modal video understanding benchmark. |
| VideoGPT+      | arXiv 2024   | Text: Phi-3-Mini-3.8B                | MLP                                              | Enhanced video understanding.                  |
| EmoLLM         | arXiv 2024   | Text: Vicuna-v1.5, Audio: Whisper    | Multi-perspective visual projection               | Multimodal emotional understanding with improved reasoning. |
| ShareGPT4Video | arXiv 2024   | Text: Mistral-7B-Instruct-v0.2       | MLP                                              | Precise and detailed video captions with hierarchical prompts. |

</details>

### EVA-CLIP ViT

<details>
<summary>Click to expand Table 3</summary>

| Model          | Venue        | Other modality encoders               | Interaction / Fusion mechanism                    | Description                                     |
|----------------|--------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------|
| VideoChat      | arXiv 2023   | Text: StableVicuna, Audio: Whisper   | Q-Former bridges visual features to LLMs for reasoning | Chat-centric model for video analysis.         |
| VAST           | NeurIPS 2023 | Text: BERT, Audio: BEATs             | Cross-attention layers                            | Omni-modality foundational model.              |
| GPT4Video      | ACMMM 2023   | Text: LLaMA 2                        | Transformer-based cross-attention layer           | Video understanding with LLM-based reasoning.  |
| VTG-LLM        | arXiv 2024   | Text: LLaMA-2-7B                     | Projection layer                                  | Enhanced video temporal grounding.             |
| AutoAD III     | CVPR 2024    | Text: GPT-3.5-turbo                  | Shared Q-Former                                   | Video description enhancement with LLMs.       |
| MiniGPT4-Video | arXiv 2024   | Text: LLaMA 2                        | Concatenates visual tokens and projects into LLM space | Video understanding with visual-textual token interleaving. |
| MA-LMM         | CVPR 2024    | Text: Vicuna                         | A trainable Q-Former                               | Memory-augmented large multimodal model.       |
| VideoLLaMA 2   | arXiv 2024   | Text: LLAMA 1.5, Audio: BEATs        | Spatial-Temporal Convolution (STC) connector      | Advancing spatial-temporal modeling and audio understanding. |
| VideoLLM-online| CVPR 2024    | Text: Llama-2-Chat/Llama-3-Instruct  | MLP projector                                     | Online video large language model for streaming video. |

</details>

### BLIP-2 ViT

<details>
<summary>Click to expand Table 4</summary>

| Model          | Venue        | Other modality encoders               | Interaction / Fusion mechanism                    | Description                                     |
|----------------|--------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------|
| LAVAD          | CVPR 2024    | Text: Llama-2-13b-chat               | Converts video features into textual prompts for LLMs | Training-free video anomaly detection using LLMs. |

</details>

### SigLIP

<details>
<summary>Click to expand Table 5</summary>

| Model          | Venue        | Other modality encoders               | Interaction / Fusion mechanism                    | Description                                     |
|----------------|--------------|---------------------------------------|--------------------------------------------------|-------------------------------------------------|
| SliME          | arXiv 2024   | Text: LLaMA3-8B                      | MLP & query transformer                          | High-resolution multimodal model for visual reasoning tasks. |
| Holmes-VAD     | arXiv 2024   | Text: LLaMA3-Instruct-70B            | Temporal sampler                                 | Multimodal LLM for video anomaly detection.    |

</details>

## Models with Pretrained Video Encoder

### Traditional video encoders

<details>
<summary>Click to expand Table 6</summary>

| Model              | Venue     | Other modality encoders | Interaction / Fusion mechanism | Description                      |
|---------------------|-----------|--------------------------|---------------------------------|----------------------------------|


</details>

### Transformer-based

<details>
<summary>Click to expand Table 7</summary>

| Model              | Venue     | Other modality encoders | Interaction / Fusion mechanism | Description                      |
|---------------------|-----------|--------------------------|---------------------------------|----------------------------------|
| LaViLa         | CVPR 2022    | Text: 12-layer Transformer           | Cross-attention modules                          | Large-scale language model.                    |

</details>

### Advanced video encoders

<details>
<summary>Click to expand Table 8</summary>

| Model              | Venue     | Other modality encoders | Interaction / Fusion mechanism | Description                      |
|---------------------|-----------|--------------------------|---------------------------------|----------------------------------|


</details>


