from tqdm.notebook import tqdm

from transformers import (
    LayoutLMTokenizer,
)
import torch
from datasets import load_dataset, Features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LayoutLMTokenizer.from_pretrained(
   "microsoft/layoutlm-base-uncased",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMForTokenClassification.from_pretrained("/content/drive/MyDrive/ReactNative/saved_model")
model.to(device)

#INFERENCE

def predict_layoutLM(img_data):
  end_boxes = []
  tokens_len = []
  end_preds = []
  actual_len = 0
  for word, box in zip(img_data["words"],img_data["bboxes"]):
    tokenized_word = tokenizer.tokenize(word)
    print("Word: ",word, " | ","Tokenized word: ",tokenized_word)
    tokens_len.append(len(tokenized_word))
    end_boxes.extend([box]*len(tokenized_word))

  num_special_tokens = 2
  if len(end_boxes) > 512 - num_special_tokens:
    end_boxes = end_boxes[:(512-num_special_tokens)]

  end_boxes = [[0,0,0,0]] + end_boxes + [[1000,1000,1000,1000]]
  encoding = tokenizer(" ".join(img_data["words"]),return_tensors="pt")


  #If sentence was smaller than max_length then it have to be padding added to final_box and final_labels to keep compatibility
  pad_token_box = [0, 0, 0, 0]
  padding_length = 512 - len(tokenizer(' '.join(img_data["words"]), truncation=True)["input_ids"])
  end_boxes += [pad_token_box] * padding_length
  print(len(end_boxes),"endoding",encoding["input_ids"][0][:512].unsqueeze(0).size())

  end_boxes = torch.tensor([end_boxes]).to(device)

  outputs = model(
    input_ids=encoding["input_ids"][0][:512].unsqueeze(0).to(device),
    bbox=end_boxes,
    attention_mask=encoding["attention_mask"][0][:512].unsqueeze(0).to(device),
    token_type_ids=encoding["token_type_ids"][0][:512].unsqueeze(0).to(device),
  )

  preds = torch.nn.functional.softmax(outputs.logits,dim=2).cpu().detach().numpy()
  preds_idx = preds.argmax(axis=2)[0]

  ## get actual predictions
  for i,token_len in enumerate(tokens_len):
    # print("Dlugosc: ",token_len, " suma: ", actual_len, " iteracja: ",i+1)
    actual_len += token_len
    if(actual_len <=512):
      end_preds.append(preds_idx[actual_len-1])

  return (preds,preds_idx,end_preds, tokens_len)

# (preds,preds_idx,end_preds,tokens_len) = predict_layoutLM(real_data_image)

# id2label =  {0:"other",1:"product",2:"price",3:"shop",4:"sum_price",5:"date" }
# label_preds = [id2label[pred] for pred in end_preds]