#labeling

label2id = {"other": 0,"product":1,"price":2,"shop":3,"sum_price":4,"date":5 }

def change_labels(data_end_format):
    for i,el in enumerate(data_end_format['labels']):
        for j,element in enumerate(el):
            data_end_format["labels"][i][j] = label2id[element]


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
    
def final_data_preparation(data,max_length=512):
  final_bbox = []
  final_labels = []
  for word,box,label in zip(data["words"],data["bboxes"],data["labels"]):
    tokenized_word = tokenizer.tokenize(word)
    # print("TOK_WORD",tokenized_word)
    #Prepare final bbox
    final_bbox.extend([box]*len(tokenized_word))
    # print("final_bbox",final_bbox)

    #Prepare final labels
    final_labels.append(label)
    # final_labels.extend([-1 for _ in range(len(tokenized_word)-1)]) # ??
    final_labels.extend([label for _ in range(len(tokenized_word)-1)])

  num_special_tokens = 2
  if len(final_bbox) > max_length - num_special_tokens:
    final_bbox = final_bbox[:(max_length-num_special_tokens)]
    final_labels = final_labels[:(max_length-num_special_tokens)]

  #Lastly make usage of special tokens in bbox and labels
  final_bbox =  [[0, 0, 0, 0]] + final_bbox + [[1000, 1000, 1000, 1000]]
  final_labels =  [0] + final_labels + [0]

  #Create encoding to get - input_ids, attention_mask, token_type_ids
  encoding = tokenizer(" ".join(data["words"]),padding="max_length",truncation=True)
  # print("ENCODING",encoding)
  # print("JOIN"," ".join(data["words"]))

  #If sentence was smaller than max_length then it have to be padding added to final_box and final_labels to keep compatibility
  pad_token_box = [0, 0, 0, 0]
  padding_length = max_length - len(tokenizer(' '.join(data["words"]), truncation=True)["input_ids"])
  final_bbox += [pad_token_box] * padding_length
  final_labels += [0] * padding_length


  ## Fill encoding with previously prepared bboxes and labels
  encoding['bboxes'] = final_bbox
  encoding['ner_tags'] = final_labels

  return encoding