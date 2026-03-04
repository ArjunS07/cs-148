You are in a directory which contains code for project 2 and project 3. Project 2 involved training a CNN on 'adversarial' MNIST data - 10k images intentionally constructed by many humans to create aderversarial examples for a digit classifier. 

Project 2 used Resnet-18 with an aggressive augmentation pipeline to achieve 95% accuracy. The final model is stored in project2/checkpoints/run10_final_9k/best_model.pt. We now have to use a ViT for project 3.


I will give you a list of features I want to implement. I will then give instructions on directory setup.

## Features


Augmentations: for the CNN, we had 
```
def get_train_transform(img_size: int = 128):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

We're going to change this to:
```
def get_train_transform(img_size: int = 128):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),  # CutOut is just larger-scale erasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

Remember val transform should be just
```
    """Deterministic transform for validation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

We're additionally going to implement _repeated augmentation_: in every epoch, sample each image multiple times with different augmentation draws, rather than once, so the model sees far more variation per unique sample. Do this by repeating each index in the dataset N times per epoch via a custom sampler. N should be 3 by default.

Additionally apply, dependent on a CLI arg, Locality Self-Attention, which contains:
1. diagonal masking, which prevents each token from attending to itself (since self-similarity is trivially high and dominates attention scores, masking it forces the model to attend to other patches)
2. learnable temperature scaling of the softmax, which allows the model to learn how sharply or diffusely to distribute attention rather than fixing it with the standard √d normalization.

Apply, dependent on a CLI arg, Shifted Patch Tokenization. This involves taking the image, creating four diagonal shifts of it (up-left, up-right, down-left, down-right), concatenating these shifted copies channel-wise with the original, and then extracting patches from this enriched representation. The effect is that each patch token now contains information about its immediate neighbors at embedding time, effectively increasing the receptive field without changing the attention mechanism. Note that this means that SPT does not change N. we still have the same number of patches after tokenization. What changes is the input dimensionality to the linear patch embedding layer: instead of projecting a patch of shape (3 x  P x  P), you're projecting a concatenated patch of shape (3x 5 x  P x  P) — the original plus four shifted copies, each with 3 channels.

Let's keep synthetic data to 3k images in default CLI args. 

### Training strategy

Learning rate warmup: 15 warmup epochs by default for 50 training epochs. Set this as an arg
Regularization:
- Stochastic depth - randomly drop entire transformer blocks. Set to 0.1


### Distillation approaches

We have a pretrained CNN model which achieves 95% accuracy in project2/checkpoints/run10_final_9k/best_model.pt. We follow approaches provided by https://arxiv.org/pdf/2012.12877 for distilling a CNN to a ViT


1. Soft distillation
Loss = (1 - \lambda) * cross entropy loss of student against ground truth+ lambda * tau^2 KL (softmax(student logics / tau), softmax(teacher logits / tau))    
where tau is temperature

2. Hard distillation
0.5 * cross entropy loss of student against ground truth + 0.5* cross entropy loss against hard decision of teacher 

Distillation token:
> We now focus on our proposal, which is illustrated in Figure 2. We add a new token, the distillation token, to the initial embeddings (patches and class token). Our distillation token is used similarly as the class token: it interacts with other embeddings through self-attention, and is output by the network after the last layer. Its target objective is given by the distillation component of the loss. The distillation embedding allows our model to learn from the output of the teacher, as in a regular distillation, while remaining complementary to the class embedding.
So when using the distillation token with hard distillation, the loss is just 0.5 CE (CLS head output, true label) + 0.5 CE (distillation head output, teacher argmax)

We should design our training pipeline such that we can switch from:
1. No distillation loss
2. Soft distillation loss with lambda and tau as hyperparameters
3. Hard distillation loss 
4. Hard distillation loss with the distillation token


distillation token is only used with hard distillation

## Directory setup

I want to follow a similar setup as project2/.
`src`: contains
- dataset.py
- model.py
- train.py
- pipeline.py (for final submission to HF - look at project 2 as a reference)

In model.py, set up a skeleton for the ViT, but don't fill it in yet - I want to code up the blocks. Let's use DeiT-tiny for the skeleton, with depth=12, dim=192, heads=3, patch_size=16.

`src` also contains visualize_augmentations, visualize_embeddings, visualize_errors, visualize_pca, verify_pipeline.py. It should also contain generate_dashboard, generate_index. Look at the original ones and make whichever changes are necessary, but many should not be required.

A major change from project 2 is that I now want to do training in a notebook so I can use colab resources. I want to use the Visual Studio code extension for Colab: https://github.com/googlecolab/colab-vscode. But you're going to need to design this in a special way - for example, cloning the repository and pulling everything it needs from project 3. 

I want scripts / a notebook to be set up for the following experiments:
1. Run a vanilla ViT, with augmentations but without distillation, SPT, LSA. 
2. Ablation over soft distillation, hard distillation, hard distillation with distillation token. Set a reasonable set of parameters for tau and lambda considering our dataset size and model size (11M params for Resnet-18)
3. Ablation over vanilla + SPT, vanilla + LSA, vanilla + SPT + LSA
4. Final script which will contain all the things we want - for now, assume we add in SPT, LSA, hard distillation with distillation token
