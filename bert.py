import torch
import json
import pickle
import bertFuncs
import ast
import numpy as np
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, \
  MultipleChoiceModelOutput
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertModel,\
  BertPreTrainedModel, AutoModelForSequenceClassification,AutoModelForTokenClassification




class BertForQuestionAnsweringPolar(BertPreTrainedModel):
  _keys_to_ignore_on_load_unexpected = [r"pooler"]

  def __init__(self, config, w_inv):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config, add_pooling_layer=False)
    ### Custom ###
    for k, v in self.bert.named_parameters():
      v.requires_grad = False

    self.w_inv = w_inv
    self.hidden_size = w_inv.shape[0]

    self.qa_outputs = nn.Linear(self.hidden_size, config.num_labels)
    ### \Custom ###

    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      start_positions=None,
      end_positions=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""
    start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    sequence_output = outputs[0]  # word vectors

    ### Custom ###
    sequence_output = torch.matmul(self.w_inv, sequence_output.permute(0, 2, 1))  # .size()
    sequence_output = sequence_output.permute(0, 2, 1)
    ### \Custom ###

    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    total_loss = None
    if start_positions is not None and end_positions is not None:
      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      ignored_index = start_logits.size(1)
      start_positions = start_positions.clamp(0, ignored_index)
      end_positions = end_positions.clamp(0, ignored_index)

      loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2

    if not return_dict:
      output = (start_logits, end_logits) + outputs[2:]
      return ((total_loss,) + output) if total_loss is not None else output

    return QuestionAnsweringModelOutput(
      loss=total_loss,
      start_logits=start_logits,
      end_logits=end_logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )


class BertForSequenceClassificationPolar(BertPreTrainedModel):
  def __init__(self, config, w_inv, softmax_bool=False, ):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config

    self.bert = BertModel(config)
    for k, v in self.bert.named_parameters():  ##Custom
      v.requires_grad = False
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    ### Custom ###
    self.w_inv = w_inv
    self.hidden_size = w_inv.shape[0]
    self.classifier = nn.Linear(self.hidden_size, config.num_labels)
    self.softmax_bool = softmax_bool
    if softmax_bool:
      self.softmax = nn.Softmax(-1)
    ### \Custom ###
    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    pooled_output = outputs[1]  # CLS token
    ### Custom ###
    pooled_output = torch.matmul(pooled_output, self.w_inv.permute(1, 0))  # .size() .transpose(0, 1)
    ### \Custom ###

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    # custom
    if self.softmax_bool:
      logits = self.softmax(logits)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
          self.config.problem_type = "single_label_classification"
        else:
          self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )

  def embedding(self, input_ids=None, attention_mask=None, token_type_ids=None):
    outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask)

    # pooled_h_states = torch.mean(outputs[0], dim=1)  # Averaged word vectors
    cls_tok_state = outputs[1]  # CLS token
    ### Custom ###
    if self.w_inv is not None:
      pooled_output = torch.matmul(cls_tok_state, self.w_inv.permute(1, 0))

    return pooled_output


class BertForMultipleChoicePolar(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.bert = BertModel(config)
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    self.classifier = nn.Linear(config.hidden_size, 1)  # config.hidden_size = 768 num_labels": 4,

    # Initialize weights and apply final processing
    self.post_init()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
        num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
        `input_ids` above)
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    inputs_embeds = (
      inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
      if inputs_embeds is not None
      else None
    )

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    reshaped_logits = logits.view(-1, num_choices)

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(reshaped_logits, labels)

    if not return_dict:
      output = (reshaped_logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return MultipleChoiceModelOutput(
      loss=loss,
      logits=reshaped_logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )


class BertForTokenClassificationPolar(BertPreTrainedModel):
  r"""
      **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
          Labels for computing the token classification loss.
          Indices should be in ``[0, ..., config.num_labels - 1]``.

  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
      **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
          Classification loss.
      **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
          Classification scores (before SoftMax).
      **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
          list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
          of shape ``(batch_size, sequence_length, hidden_size)``:
          Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      **attentions**: (`optional`, returned when ``config.output_attentions=True``)
          list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      model = BertForTokenClassification.from_pretrained('bert-base-uncased')
      input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
      labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
      outputs = model(input_ids, labels=labels)
      loss, scores = outputs[:2]

  """

  def __init__(self, config, freeze_bool=True, w_inv=None, softmax_bool=False):
    super().__init__(config)
    # super(BertForTokenClassification, self).__init__(config)
    self.num_labels = config.num_labels
    self.bert = BertModel(config)
    if freeze_bool:
      for k, v in self.bert.named_parameters():  ##Custom
        v.requires_grad = False
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    self.w_inv = w_inv
    if self.w_inv is not None:
      self.hidden_size = w_inv.shape[0]
    else:
      self.hidden_size = config.hidden_size
    self.classifier = nn.Linear(self.hidden_size, config.num_labels)
    self.init_weights()

    if softmax_bool:
      self.softmax = nn.Softmax(-1)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None,
              position_ids=None, head_mask=None, labels=None):
    outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask)

    sequence_output = outputs[0]  # word vectors
    ### Custom ###
    sequence_output = torch.matmul(self.w_inv, sequence_output.permute(0, 2, 1))  # .size()
    sequence_output = sequence_output.permute(0, 2, 1)
    ### \Custom ###
    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
      else:
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), scores, (hidden_states), (attentions)
















def get_word_idx(sent: str, word: str):
  try:
    return sent.split(" ").index(word)
  except:
    print(word)
    print(sent)
    return 0


def get_hidden_states(encoded, token_ids_word, model):
  #From: https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
  """Push input IDs through model. Stack layers.
     Select only those subword token outputs that belong to our word of interest
     and average them."""
  with torch.no_grad():
    #print(encoded)
    output = model(**encoded)


  # Get all hidden states, dim 13 x #token x 768
  states = output.hidden_states
  # Select only the second to last layer by default, dim #token x 768
  output = states[-2][0]
  # Only select the tokens that constitute the requested word
  word_tokens_output = output[token_ids_word]
  return word_tokens_output.mean(dim=0) # dim 768



def forward1Word(tokenizer, model, sentence, word):
  idx = get_word_idx(sentence, word)  # position of the antonym in the sentence. Ex: 2
  encoded = tokenizer.encode_plus(sentence, return_tensors="pt")
  # get all token idxs that belong to the antonym
  token_ids_word = np.where(np.array(encoded.word_ids()) == idx)  # Ex:(array([3, 4, 5]),)

  # forward the sentence and get embedding of the cur word:
  embedding = get_hidden_states(encoded, token_ids_word, model)

  return embedding

def loadAntonymsFromJson(dict_path):
  ## This function reads the antonyms and their example sentences from a json

  if "txt" in dict_path:
    with open(dict_path) as f:
      antonym_dict = json.load(f)

  return antonym_dict


def createPolarDimension(model, tokenizer, out_path, antonym_path=""):
  print("Start forwarding the Polar oposites ...")
  if antonym_path == "":
    dict_path = "antonyms/antonym_wordnet_example_sentences_readable_extended.txt" #antonym_wordnet_example_sentences_readable_extended
  else:
    dict_path = antonym_path
  antonym_dict = loadAntonymsFromJson(dict_path)
  debug_print = False
  direction_vectors=[]
  debug_counter = 0

  for antonymString, sentences in antonym_dict.items():
    antonym_names = list(sentences.keys())
    if len(antonym_names) == 1:
      print(antonym_names)


  for antonymString, sentences in antonym_dict.items():
    #if debug_counter < 2:
      # "['disentangle', 'entangle']" to ['disentangle', 'entangle']
      antonym_list = ast.literal_eval(antonymString)

      # Get Antonym names and their example sentences:
      antonym_names=list(sentences.keys())
      antonym1_name=antonym_names[0]
      antonym2_name = antonym_names[1]

      antonym1_sentences = sentences[antonym1_name]
      antonym2_sentences = sentences[antonym2_name]

      # left antonym
      #iterate over the example sentences and average the word-embedding
      cur_word = antonym1_name.split(" ")  # Ex: [disentangle], or [put, in]
      ant1_embedding_list= []
      for example_sentence in antonym1_sentences:
        ant1_wordpart_list=[]
        for antonym_part in cur_word:
          cur_sent=example_sentence #Ex: Can you disentangle the cord?
          cur_embedding=forward1Word(tokenizer, model, cur_sent, antonym_part)
          ant1_wordpart_list.append(cur_embedding)
        cur_anti_embedding = torch.mean(torch.stack(ant1_wordpart_list), dim=0)
        if not True in torch.isnan(cur_anti_embedding):
          ant1_embedding_list.append(cur_anti_embedding)
      ant1_embedding= torch.mean(torch.stack(ant1_embedding_list), dim=0).numpy()
      if len(ant1_embedding) != 768 and len(ant1_embedding) != 1024:
        print(len(ant1_embedding))
        print(cur_word)


      # right antonym
      # iterate over the example sentences and average the word-embedding
      cur_word = antonym2_name.split(" ")  # Ex: [disentangle], or [put, in]
      ant2_embedding_list = []
      for example_sentence in antonym2_sentences:
        ant2_wordpart_list = []
        for antonym_part in cur_word:
          cur_sent = example_sentence  # Ex: Can you disentangle the cord?
          cur_embedding = forward1Word(tokenizer, model, cur_sent, antonym_part)
          ant2_wordpart_list.append(cur_embedding)
        cur_anti_embedding = torch.mean(torch.stack(ant2_wordpart_list), dim=0)
        if not True in torch.isnan(cur_anti_embedding):
          ant2_embedding_list.append(cur_anti_embedding)
        else:
          print(example_sentence)
      ant2_embedding = torch.mean(torch.stack(ant2_embedding_list), dim=0).numpy()
      if len(ant2_embedding) != 768 and len(ant2_embedding) != 1024:
        print(len(ant2_embedding))
        print(cur_word)


      cur_direction_vector=ant2_embedding - ant1_embedding
      cur_direction_vector_numpy= cur_direction_vector
      if np.isnan(np.min(cur_direction_vector_numpy)):
        print("Nan....")
        print(antonym1_sentences)
        for sentence in ant1_embedding_list:
          print(sentence)
        print(np.min(ant1_embedding))
        print(antonym2_sentences)
        print(np.min(ant2_embedding))

      antonym_dict[antonymString]["direction"] = cur_direction_vector_numpy
      direction_vectors.append(cur_direction_vector_numpy)


  safetyBool=True
  if safetyBool:
    out_dir_path=out_path+"polar_dimensions.pkl"
    with open(out_dir_path, 'wb') as handle:
      pickle.dump(direction_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    print("Enable safety bool to dump!")









def initCustomModel(original_model, new_model_path, task):

  # Load original Model to initilize the new (polar) model
  print("Load original Model to initilize the new (polar) model")
  tokenizer = AutoTokenizer.from_pretrained(original_model)

  if task == "QuestionAnswering":
    base_model=AutoModelForQuestionAnswering.from_pretrained(original_model, output_hidden_states=True)
    base_model = base_model.bert
  elif task =="SequenceClassification":
    base_model=AutoModelForSequenceClassification.from_pretrained(original_model, output_hidden_states=True,ignore_mismatched_sizes=True)
    base_model=base_model.bert
  elif task == "BertForTokenClassification":
    base_model = AutoModelForTokenClassification.from_pretrained(original_model, output_hidden_states=True)
    base_model = base_model.bert
    print("Token Classification")
  else:
    print("Please choose a correct Bert-Task")



  # Create (and dump) Polar space with the bert-part of the model
  print("Create (and dump) Polar space with the bert-part of the model")
  antonym_path=""
  createPolarDimension(model=base_model, tokenizer=tokenizer, out_path=new_model_path, antonym_path=antonym_path)

  # Load Polar space
  antonym_path_wordnet = new_model_path + "polar_dimensions.pkl"#"antonym_wordnet_dir_matrix.pkl"
  W, W_inverse = bertFuncs.getW(antonym_path_wordnet)
  W_inverse = torch.from_numpy(W_inverse)

  # Initilize new Polar model
  if task == "QuestionAnswering":
    class_model = BertForQuestionAnsweringPolar.from_pretrained(original_model, w_inv=W_inverse,
                                                                output_hidden_states=True,
                                                                ignore_mismatched_sizes=True)
  elif task =="SequenceClassification":
    class_model = BertForSequenceClassificationPolar.from_pretrained(original_model, w_inv=W_inverse,
                                                                output_hidden_states=True,
                                                                ignore_mismatched_sizes=True)
  elif task == "BertForTokenClassification":
    class_model = BertForTokenClassificationPolar.from_pretrained(original_model, w_inv=W_inverse,
                                                                     freeze_bool=True,
                                                                     output_hidden_states=True,
                                                                     ignore_mismatched_sizes=True,
                                                                     softmax_bool=True)
  else:
    print("Please choose a correct Bert-Task")

  # Save and dump new Polar model
  class_model.save_pretrained(new_model_path)
  tokenizer.save_pretrained(new_model_path)

  print("Computing average embedding")
  #compNormalizationTerm(class_model.bert, tokenizer, new_model_path)
  #TODO

  print(class_model)





if __name__ == "__main__":
  task = "stsb"
  original_model="downstream_tasks/GLUE/models/"+str(task)+"/Baseline/"
  new_model_path="downstream_tasks/GLUE/models/"+str(task)+"/Polar/"
  import os
  if not os.path.exists(new_model_path):
    os.makedirs(new_model_path)

  task="SequenceClassification"
  initCustomModel(original_model, new_model_path, task)









