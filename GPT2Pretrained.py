from tokenise import BPE_token
from pathlib import Path
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os

class GPT2Pretrained:
    
    def __init__(self, pathBooks):
        self.pathBooks = pathBooks
        self.stringTokenized = ""
        self.generatedText = ""
        self.path = []
        self.save_path = 'TokenizedData'
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def Token(self):
        Books = Path(self.pathBooks).glob("**/*.txt")
        self.paths = [str(book) for book in Books]
        self.tokenizer = BPE_token()
        self.tokenizer.bpe_train(self.paths)
        self.tokenizer.save_tokenizer(self.save_path)
        print(f'Your tokenized data is saved in the "{self.save_path}" folder.')
        
    def CreationGPT2(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.save_path)
        self.tokenizer.add_special_tokens({"eos_token": "</s>",
                                           "bos_token": "<s>",
                                           "unk_token": "<unk>",
                                           "pad_token": "<pad>",
                                           "mask_token": "<mask>"})
        config = GPT2Config(vocab_size = self.tokenizer.vocab_size, 
                            bos_token_id = self.tokenizer.bos_token_id, 
                            eos_token_id = self.tokenizer.eos_token_id)
        self.model = TFGPT2LMHeadModel(config)
        
    def EncodeTokenizer(self):
       SingleString = ''
       for filename in self.paths:
           with open(filename, "r", encoding='utf-8') as file:
               SingleString += file.read() + "</s>"
               file.close()
       self.stringTokenized = self.tokenizer.encode(SingleString)
       
    def OptimizerGPT2(self):
        Optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        Metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=Optimizer, loss=[Loss, *[None] * self.model.config.n_layer], metrics=[Metric])
    
    def DataSetFit(self):
        Inputs, Labels, Examples = [], [], []
        BlockSize = 100
        BatchSize = 12
        BufferSize = 1000
        
        for value in range(0, len(self.stringTokenized) - BlockSize + 1, BlockSize):
            Examples.append(self.stringTokenized[value:value + BlockSize])
            
        for example in Examples:
            Inputs.append(example[:-1])
            Labels.append(example[1:])
            
        self.dataset = tf.data.Dataset.from_tensor_slices((Inputs, Labels))
        self.dataset = self.dataset.shuffle(BufferSize).batch(BatchSize, drop_remainder=True)
        
    def FitGPT2(self, numEpoch = 10):
        self.model.fit(self.dataset, epochs = numEpoch)
        
        
    def SaveGPT2(self ,pathSaveModel = './model_bn_custom/'):
        if not os.path.exists(pathSaveModel):
            os.mkdir(pathSaveModel)
        ModelSave = self.model.module if hasattr(self.model, 'module') else self.model
        os.path.join(pathSaveModel, WEIGHTS_NAME)
        SaveModelConfig = os.path.join(pathSaveModel, CONFIG_NAME)
        self.model.save_pretrained(pathSaveModel)
        ModelSave.config.to_json_file(SaveModelConfig)
        self.tokenizer.save_pretrained(pathSaveModel)
    
    def LoadGPT2(self, pathLoadModel = './model_bn_custom/'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(pathLoadModel)
        self.model = TFGPT2LMHeadModel.from_pretrained(pathLoadModel)
    
    def TextGenerator(self, text, maxLength = 100,numBeams = 5, temperature = 0.7, noRepeatNgramSize = 2, numReturnSequences = 5 ):
        InputIds = self.tokenizer.encode(text, return_tensors='tf')
        Output = self.model.generate(InputIds,
                                     max_length = maxLength, 
                                     num_beams = numBeams,
                                     temperature = temperature,
                                     no_repeat_ngram_size = noRepeatNgramSize,
                                     num_return_sequences = numReturnSequences)
        self.generatedText = self.tokenizer.decode(Output[0])
        print(self.generatedText)
        