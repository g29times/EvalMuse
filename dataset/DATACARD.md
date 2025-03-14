# Dataset Card for EvalMuse
## Dataset Structure
The repository includes images.zip.part-{aa-af}, which you need to merge manually. Use the following command to combine them:

```
cat images.zip.part-* > images.zip
unzip images.zip
```

train_list.json: data used for training, containing detailed fine-grained annotations.
test.json: data used for testing, annotations will be available later.
prompt_t2i.json: selected 200 prompts and corresponding elements for image-text alignment evaluation of t2i models.
split confidence , 1 means gpt based element split may got wrong result, while 0 stands for the gpt split the elements correctly.
attribute confidence, 0 stands for the annotator was not sure about does the attribute keyword in the prompt match the corresponding image, since some attribute terms are hard to recognize for normal annotators.
prompt meaningless, 1 stands for the prompt does not have major subject or the prompt is too abstract to evaluate text-image alignment.