

# EvalMuse-40K

**EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation**

<p align="center">
  <img src="assets/images/framework.png" alt="EvalMuse Logo">
</p>

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
   git clone https://github.com/your-username/EvalMuse-40K.git
   cd EvalMuse-40K
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset and Preprocess the Data**:
   ```bash
      sh scripts/download.sh
   ```
4. **Run the Training Scripts**:
   ```bash
      sh scripts/train.sh
   ```
5. **Evaluate the Models**:
   ```bash
      # You can download the pre-trained models from HuggingFace:
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

If you use EvalMuse-40K in your research, please cite our paper:
