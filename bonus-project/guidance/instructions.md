You are in a subdirectory `bonus_project` of a directory `cs-148` containing work for a deep learning class. Ignore the `hw` folders.

Projects 2, 3, and 4 contain the setup for training deep neural network classifiers on an extremely adversarial dataset, which has been copied to data/dataset here.

The goal of this final subproject is to run 3 experiments to produce 3 plots:
1. Plot 1: Log Error vs. Log Sample countHow did the number of total training samples change the validation and training error?
2. Plot 2: Log Error vs. Log Parameter countHow does the error scale with the number of parameters?
3. Plot 3: Log Error vs. Wall ClockHow did the error scale with inference time? Use the same compute resource for all checkpoints,

My plan is to extract the CNN and ViT setups and re-run training experiments with different # sample counts and diferent depths. 

I will come up with a plan plot 3 later. 

project2 contains the setup

├── bonus-project
│   └── instructions.md
├── hw1
│   ├── arjunsharma_cs148a_hw1.aux
│   ├── arjunsharma_cs148a_hw1.log
│   ├── arjunsharma_cs148a_hw1.pdf
│   ├── arjunsharma_cs148a_hw1.synctex.gz
│   ├── arjunsharma_cs148a_hw1.tex
│   ├── dataset
│   ├── examples
│   ├── fashion_mnist.zip
│   ├── figs
│   ├── grad
│   ├── grad.egg-info
│   ├── HW1.pdf
│   ├── pyproject.toml
│   ├── requirements_dev.txt
│   ├── requirements.txt
│   ├── results.json
│   ├── run_tests.py
│   ├── SETUP.md
│   ├── uv.lock
│   └── zip_assignment.py
├── hw2
│   ├── arjunsharma_cs148a_hw2.aux
│   ├── arjunsharma_cs148a_hw2.log
│   ├── arjunsharma_cs148a_hw2.pdf
│   ├── arjunsharma_cs148a_hw2.synctex.gz
│   ├── arjunsharma_cs148a_hw2.tex
│   └── HW2.pdf
├── hw3
│   ├── arjunsharma_cs148a_hw3.aux
│   ├── arjunsharma_cs148a_hw3.log
│   ├── arjunsharma_cs148a_hw3.pdf
│   ├── arjunsharma_cs148a_hw3.synctex.gz
│   ├── arjunsharma_cs148a_hw3.tex
│   └── code
├── hw4
│   ├── arjunsharma_cs148a_hw4.aux
│   ├── arjunsharma_cs148a_hw4.log
│   ├── arjunsharma_cs148a_hw4.pdf
│   ├── arjunsharma_cs148a_hw4.synctex.gz
│   └── arjunsharma_cs148a_hw4.tex
├── project1a
│   ├── adversarial_mnist
│   ├── asharma3_report.aux
│   ├── asharma3_report.bbl
│   ├── asharma3_report.blg
│   ├── asharma3_report.log
│   ├── asharma3_report.out
│   ├── asharma3_report.pdf
│   ├── asharma3_report.synctex.gz
│   ├── asharma3_report.tex
│   ├── figs
│   ├── refs.bib
│   └── venv
├── project1b
│   ├── asharma3_project1b_cs148.pdf
│   └── asharma3_project1b_writeup.md
├── project2
│   ├── __pycache__
│   ├── checkpoints
│   ├── colab_training.ipynb
│   ├── data
│   ├── logs
│   ├── pipeline-cnn.pt
│   ├── report
│   ├── src
│   ├── submission.json
│   ├── test_pipeline.py
│   └── uv.lock
├── project3
│   ├── checkpoints
│   ├── data
│   ├── exp1_vanilla.ipynb
│   ├── exp2_distillation.ipynb
│   ├── exp3_spt_lsa.ipynb
│   ├── exp4_final.ipynb
│   ├── guidance.md
│   ├── pipeline-vit.pt
│   ├── pyproject.toml
│   ├── report
│   ├── src
│   ├── submission.json
│   └── uv.lock
└── project4
    ├── checkpoints
    ├── count_params.py
    ├── CS148a_proj4_FM_starter.ipynb
    ├── data
    ├── embeddings
    ├── guidance
    ├── logs
    ├── main.py
    ├── pyproject.toml
    ├── report
    ├── src
    └── uv.lock
