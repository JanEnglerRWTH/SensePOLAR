import numpy as np
import torch
import pickle
from transformers import BertTokenizerFast, BertModel
from scipy import linalg



def getBert():
  """ Returns the pretrained (base uncased) bert model that is used as the standard WE model, with its tokenizer"""
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
  model.eval()
  return tokenizer, model


def get_word_idx(sent: str, word: str):
  return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model):
  #From: https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
  """Push input IDs through model. Stack layers.
     Select only those subword token outputs that belong to our word of interest
     and average them."""
  with torch.no_grad():
    output = model(**encoded)
  # Get all hidden states, dim 13 x #token x 768 in base model
  states = output.hidden_states
  # Select only the second to last layer by default, dim #token x 768
  output = states[-2][0]
  # Only select the tokens that constitute the requested word
  word_tokens_output = output[token_ids_word]
  return word_tokens_output.mean(dim=0) # dim 768, average the token embeddings


def forwardWord(tokenizer, model, sentence, word):
  # From: https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
  idx = get_word_idx(sentence, word)  # position of the word in the sentence. Ex: 2
  encoded = tokenizer.encode_plus(sentence, return_tensors="pt")
  # get all token idxs that belong to the antonym
  token_ids_word = np.where(np.array(encoded.word_ids()) == idx)  # Ex:(array([3, 4, 5]),)

  # forward the sentence and get embedding of the cur word:
  embedding = get_hidden_states(encoded, token_ids_word, model)

  return embedding



def getW(antonym_path):
  # returns W and W^{-1}: For base change into new space
  with open(antonym_path, 'rb') as curAntonymsPickle:
    curAntonyms = pickle.load(curAntonymsPickle)

  if len(curAntonyms[0]) == 3:
    #Case [anto-1, anto1, direction]
    axisList=[]
    for antony in curAntonyms:
      axisList.append(antony[2])
  else:
    # Case [direction1, direction2]
    print(len(curAntonyms))
    axisList=curAntonyms#[0:768] #1763 pairs
  W = np.matrix(axisList)

  #W_inverse = np.linalg.pinv(np.transpose(W),rcond=0.001)
  W_inverse = linalg.pinv(np.transpose(W))
  return W, W_inverse

def printMeaningOfWord(word_embedding, antonym_path, numberPolar, definition_path):
  # For matchind dimension to antonym pair
  with open(antonym_path, 'rb') as curAntonymsPickle:
    antonyms = pickle.load(curAntonymsPickle)
  # for retrieving wordnet definitions of the antonym pair
  with open(definition_path, 'rb') as curDefPickle:
    definitions = pickle.load(curDefPickle)

  word = word_embedding
  # sort the embedding on absolute value
  thisdict = {}
  for count, value in enumerate(word):
    thisdict[count] = value
  sortedDic = sorted(thisdict.items(), key=lambda item: abs(item[1]))
  sortedDic.reverse()

  axis_list=[]
  # Retrieve and print top-numberPolar dimensions
  for i in range(0, numberPolar):
    cur_Index = sortedDic[i][0]
    cur_value = sortedDic[i][1]

    leftPolar = antonyms[cur_Index][0][0]
    leftDefinition = definitions[cur_Index][0]

    rightPolar = antonyms[cur_Index][1][0]
    rightDefinition = definitions[cur_Index][1]

    axis=leftPolar + "---" + rightPolar
    axis_list.append(axis)

    # Print
    print("Top: ", i)
    print("Dimension: ", leftPolar + "<------>"+ rightPolar)
    print("Definitions: ", leftDefinition+ "<------>"+ rightDefinition)
    if cur_value <0:
      print("Value: " + str(cur_value))
    else:
      print("Value:                      " + str(cur_value))
    print("\n")
  return axis_list






def analyzeWord(cur_word, context, model=None,tokenizer=None, antonym_path="", normalize_term_path="antonyms/wordnet_normalize.pkl",numberPolar=5):
  """ Prints out the top POLAR_C dimensions of a given word in a given context

  Args:
      cur_word (str): The word to analyze
      context (str): The context (~example sentence) in which the word should be analyzed. HINT: Must contain the word with the exact spelling!!!
      model: The BERT model to be used. Defaults to 'bert-base-uncased'
          (default is None)
      tokenizer: The  tokenizer for the BERT model.
          (default is None)
      antonym_path (str): Path where *a*, the base-change antonym-matrix is stored. If not passed, defaults to the WordNet antonyms
      normalize_term_path (str): Path where the normalization-matrix is stored (Non-Polar Space).
      numberPolar (int): Top-numberPolar are printed and returned
  Returns:
      list: of the top-numberPolar dimensions of the word.
  """

  print("Analyzing the word: ", cur_word)
  print("In the context of: ", context)

  if cur_word not in context.split(" "):
    print("Warning:")
    print("The context must contain the *exact* word you want to analyze!")
    print("(Remove or add a space to puncturation if directly attached to the word you want to analyze.) ")
    print("Try again")
    return None


  debug_flag=False

  # get model
  if model is None:
    tokenizer, model = getBert()

  # forward the word
  cur_word_emb = forwardWord(tokenizer, model, context, cur_word)

  # Normalization
  if normalize_term_path is not None:
    import pickle
    with open(normalize_term_path, 'rb') as curAntonymsPickle:
      normalize_term = pickle.load(curAntonymsPickle)
    cur_word_emb = cur_word_emb - normalize_term


  #get polar space
  if antonym_path =="":
    antonym_path = "antonyms/antonym_wordnet_base_change.pkl"
  _, W_inv_np = getW(antonym_path)
  W_inv_torch = torch.from_numpy(W_inv_np)

  #base-change into polar space
  polar_emb = torch.matmul(W_inv_torch,cur_word_emb)
  polar_emb_np=polar_emb.numpy()


  #visulize/ print top dimensions
  antonym_path_lookup="antonyms/lookup_anto_example_dict.pkl"
  defintion_path="antonyms/lookup_synset_definition.pkl"

  axis_list=printMeaningOfWord(polar_emb_np, antonym_path_lookup, numberPolar,defintion_path) #np.max(np.abs(polar_emb_np))

  if debug_flag:
    print(cur_word_emb.shape)
    print(type(W_inv_np))
    print(type(W_inv_torch))
    print(W_inv_torch.shape)
    print(polar_emb.shape)


  return axis_list #, polar_emb_np



# bert uncased
def main():
  normalize_term = "antonyms/wordnet_normalize.pkl"
  numberPolar = 10
  word = "sleep"
  sentence = "i want to go to sleep"

  axis_list_1 = analyzeWord(word, sentence, numberPolar=numberPolar, model=None, antonym_path="", normalize_term_path=normalize_term)


if __name__ == "__main__":
  main()































