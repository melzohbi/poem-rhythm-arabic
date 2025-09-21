
from transformers import pipeline

if __name__ == "__main__":

    beat_aligned_generator = pipeline(
        "text2text-generation", model='melzohbi/byt5-b-ar')
    poem = "بادر العشر عشر كفيك <extra_id_0>1010<extra_id_1> وتمنى هلاله منك تما<extra_id_2>"
    generated_words = beat_aligned_generator(poem, max_length=30, do_sample=True,
                                             num_return_sequences=5, temperature=0.7, top_p=1)

    print(generated_words)
