CS148a Assignment 4: Standing on the Shoulders of Foundation Models
In this project, you will be leveraging embeddings from foundation models to classify
MNIST-in-the-wild. You will be evaluated on your understanding of your chosen foundation model’s
architecture and your downstream model’s training schema. There will not be a competition
component for this assignment. Instead, we will award bonus points for an additional section.
Task Overview
Using the training data collected by yourself and your classmates, build a downstream pipeline that
uses embeddings from a foundation model, and classifies images of digits (0–9) into their correct
labels. You are free to choose the depth, width, and architectural details of your model. We will
mainly focus on CLIP and DINO, with optional (extra-credit, 5%) zero-shot evaluation on Qwen.
This project checkpoint is a little different from previous models. Your goal is to take advantage of
the latent spaces of these foundation models. Remember what you learned in lecture - these large
models have already learned very rich semantically organized embeddings. Perhaps they will help
classify our MNIST-in-the-wild dataset!
Model Training
You will need a GPU to train your model. If you have your own – great! Otherwise, there are a few
ways you can get compute:
1. We are providing $50 GPU credits to each of you using Google Cloud. ☁Follow the
instructions here☁. Make sure to use a Non-Caltech gmail for this.
2. Note that Google also has free Colab Pro for students. To access GPUs that way: go to this
page with the Google account you plan to use, choose Pro for students.
a. If you ran out of compute units, we will reimburse up to $50 on Colab units for each
student. Keep your receipts, and be mindful of running out.
Here is a starter notebook ⬅⬅⬅ to get you situated. It contains instructions for loading in the
training data, loading in CLIP and DINO, training downstream classifiers, and zero shot evaluation
on foundation models. Feel free to run it in Google Colab directly (after making a copy to your own
drive). To run it on Google Cloud or on a local machine, simply download the .ipynb notebook.
Required Writeup Questions
In your writeup, answer the following:
Part 1: CLIP zero-shot run
• How does CLIP do on its own?
• What is the original training setup for CLIP?
• How many parameters are in your CLIP checkpoint?
• Are there certain things CLIP seems to miss in the training data? Visualize them, and explain why
you think CLIP didn’t classify it correctly.Part 2: CLIP downstream model
• How did you train your model? What loss function did you use, and why?
• What learning rate and batch size did you choose? How sensitive was training to these choices?
Show any figures you have.
• How long did you train for? Why did you choose that number of epochs? Show any figures you
have.
• How does your model perform during validation? Why do you believe your validation strategy is
reliable?
• Why did you make these architectural and optimization choices overall?
• How is this different from previous CNN/ViT training?
Part 3: DINO downstream model
• How many parameters does your DINO model have?
• What is the original training setup for DINO?
• How many parameters in your downstream classifier?
• How does it perform during validation? Is it better or worse than using CLIP embeddings?
• Visualize specific differences between your model from Part 2 to this model. Specifically, show
examples which are misclassified by CLIP embeddings or DINO embeddings, that are correctly
classified by the other. Why might these be working with one embedding space but not the other?
Part 4: Autoregressive VLM (Extra credit - 5%)
• How many parameters does this model have?
• What tokenizer and special tokens does the VLM use, and how do those choices affect
number-formatting and digit outputs (e.g., “7” vs “seven”)?
• What image/text processor (chat template, vision pre-processing, normalization, resizing) is used
in the Hugging Face pipeline? Why do you think these are important for model performance?
• What is the training setup for this model?
• How well does the zero-shot evalu ation do?
• Are there prompting strategies that increase or decrease the score?
• Are there certain things the VLM seems to miss in the training data? Visualize them, and explain
why you think it didn’t classify it correctly.
Part 5: Overview
• Which performed the best? Your CNN, your ViT, CLIP, CLIP-downstream, DINO-downstream, or an
autoregressive VLM (if you tested one)?
• Why might that be?
Use of AI Tools
If you use AI in this project, you must clearly document how they were used and include relevant
conversation logs in your writeup, under an “AI usage” heading.Student Deliverables
- Writeup (PDF): Single PDF (up to 6 pages) addressing all questions above, with figures/tables as
needed. ALSO INCLUDE A LINK TO YOUR FINAL NOTEBOOK/CODE BASE, uploaded to Google Drive
with view access enabled for all.
Released: 3/5/2026
Due Date: 3/18/2026
This
Version: 03052026_9:00PM
