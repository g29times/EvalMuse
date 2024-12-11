<div align="center">
   <h1>EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation</h1>
   <div>
      <a href="https://github.com/DYEvaLab/EvalMuse"><img src="https://img.shields.io/github/stars/DYEvaLab/EvalMuse"/></a>
      <a href=""><img src="https://img.shields.io/badge/Arxiv-2309:1418-red"/></a>
       <a href="https://huggingface.co/datasets/DY-Evalab/EvalMuse"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green"></a>
   </div>
      <p align="center">
      <img src="assets/images/framework.png" alt="EvalMuse framework">
   </p>
</div>


## Introduction
EvalMuse-40K is a reliable and fine-grained benchmark designed to evaluate the performance of Text-to-Image (T2I) generation models. It comprises 40,000 image-text pairs with comprehensive human annotations for image-text alignment-related tasks.

### Key Features

- **Large-scale Dataset**: Includes 40,000 image-text pairs with over 1 million fine-grained human annotations.
- **Diversity and Reliability**: Employs strategies like balanced prompt sampling and data re-annotation to ensure diversity and reliability.
- **Fine-grained Evaluation**: Categorizes elements during fine-grained annotation, allowing evaluation of specific skills at a granular level.
- **New Evaluation Methods**: Introduces FGA-BLIP2 and PN-VQA methods for end-to-end fine-tuning and zero-shot fine-grained evaluation.

## Getting Started

To use the EvalMuse-40K dataset and replicate the experiments, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DYEvaLab/EvalMuse
   cd EvalMuse
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset and Preprocess the Data**:
   ```bash
   # Download the dataset from Huggingface
   sh scripts/download.sh

   # Average the annotation scores and calculate the variance of the alignment scores for different image-text pairs corresponding to the same prompt
   python3 process/process_train.py

   # Corresponds the element split from the prompt to a specific index in the prompt
   python3 process/element2mask.py
   ```
4. **Run the Training Scripts**:
   ```bash
   sh scripts/train.sh
   ```
5. **Evaluate the Models**:

   You can download the pre-trained FGA-BLIP2 model weights from 
   [[Huggingface]()]
   or [[Baidu Cloud]()].
   ```bash
   sh scripts/eval.sh
   ```


<!-- ## Usage

The repository contains the following directories:

- `data/`: Contains the dataset files and instructions on how to preprocess them.
- `models/`: Houses the FGA-BLIP2 model implementation and pre-trained weights.
- `scripts/`: Includes the scripts for training and evaluating the models.
- `evaluation/`: Contains the PN-VQA evaluation methodology and code. -->

## Contributing

We welcome contributions to EvalMuse-40K. If you have ideas or bug reports, please open an issue or submit a pull request.


## Citation

If you find EvalMuse-40K useful for your research, please consider cite our paper:
