---
license: bsd-3-clause
dataset_info:
  features:
    - name: prompt_id
      dtype: string
    - name: prompt
      dtype: string
    - name: type
      dtype: string
    - name: img_path
      dtype: string
    - name: total_score
      sequence: int64
    - name: element_score
      dtype: string
    - name: promt_meaningless
      sequence: int64
    - name: split_confidence
      sequence: int64
    - name: attribute_confidence
      sequence: int64
    - name: fidelity_label
      sequence: string
configs:
- config_name: default
  data_files: train_list.json
---

# Dataset Card for EvalMuse

## Dataset Description

- **Repository:** https://github.com/DYEvaLab/EvalMuse
- **Paper:** EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation

## Dataset Structure

The repository includes `images.zip.part-{aa-af}`, which you need to merge manually. Use the following command to combine them:

```bash
cat images.zip.part-* > images.zip
unzip images.zip
```

- train_list.json: data used for training, containing detailed fine-grained annotations.
- test.json: data used for testing, annotations will be available later.
- prompt_t2i.json: selected 200 prompts and corresponding elements for image-text alignment evaluation of t2i models.
- split confidence , 1 means gpt based element split may got wrong result, while 0 stands for the gpt split the elements correctly.
- attribute confidence, 0 stands for the annotator was not sure about does the attribute keyword in the prompt match the corresponding image, since some attribute terms are hard to recognize for normal annotators.
- prompt meaningless, 1 stands for the prompt does not have major subject or the prompt is too abstract to evaluate text-image alignment.

## Citation

```
@misc{han2024evalmuse40kreliablefinegrainedbenchmark,
      title={EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation}, 
      author={Shuhao Han and Haotian Fan and Jiachen Fu and Liang Li and Tao Li and Junhui Cui and Yunqiu Wang and Yang Tai and Jingwei Sun and Chunle Guo and Chongyi Li},
      year={2024},
      eprint={2412.18150},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.18150}, 
}
```