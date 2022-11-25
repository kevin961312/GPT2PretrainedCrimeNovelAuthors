from transformers import  TFGPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
tokenizer = GPT2Tokenizer.from_pretrained('./model_bn_custom/')
model = TFGPT2LMHeadModel.from_pretrained('./model_bn_custom/')

text = "En una noche oscura, Hercules Poirot pudo ir a investigar la desaparición de las gemelas en el castillo."
input_ids = tokenizer.encode(text, return_tensors='tf')
beam_output = model.generate(
  input_ids,
  max_length = 100,
  num_beams = 5,
  temperature = 0.6,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
TextGenerate = [tokenizer.decode(beam_output[0]),'']
SimilarPrhase = []
with open('AllPrhase.txt', encoding = 'utf-8') as f:
    for line in f:
        TextGenerate[1] = line.strip()
        embedding = model.encode(TextGenerate, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embedding, embedding)
        score = float(cosine_scores[1][0])
        if(len(SimilarPrhase)>0 and score > SimilarPrhase[0]):
            SimilarPrhase[0] = score
            SimilarPrhase[2] = TextGenerate[1]
        else:
            SimilarPrhase.append(score)
            SimilarPrhase.append(TextGenerate[0])
            SimilarPrhase.append(TextGenerate[1])