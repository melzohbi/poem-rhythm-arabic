# Tahḏīb: A Rhythm-Aware Phrase Insertion for Classical Arabic Poetry


## Model Summary

This repository contains an implementation of a beat-aligned poetry filler model using the ByT5 transformer for Arabic poetry.


## Using Text2Text Pipeline

To get started with the model, use the following code. Note that the beat pattern should be enclosed between `<extra_id_0>` and `<extra_id_1>`.

```python
from transformers import pipeline

    beat_aligned_generator = pipeline(
        "text2text-generation", model='melzohbi/byt5-b-ar')
    poem = "بادر العشر عشر كفيك <extra_id_0>1010<extra_id_1> وتمنى هلاله منك تما<extra_id_2>"
    generated_words = beat_aligned_generator(poem, max_length=30, do_sample=True,
                                             num_return_sequences=5, temperature=0.7, top_p=1)

    print(generated_words)

```


## Citation

If you use this model in your research, please cite the following paper:

```
@inproceedings{will come later
}
```
