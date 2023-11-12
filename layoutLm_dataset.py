from torch.utils.data import Dataset, DataLoader

class LayoutLMDataset(Dataset):
    def __init__(self, data_results):#input_ids,bboxes, attention_mask, token_type_ids,labels
        self.data_results = data_results
        # self.input_ids = input_ids
        # self.bboxes=bboxes
        # self.attention_mask = attention_mask
        # self.token_type_ids = token_type_ids
        # self.labels = labels

    def __len__(self):
        return len(self.data_results)

    def __getitem__(self, idx):
        input_id = self.data_results[idx]["input_ids"]
        bbox = self.data_results[idx]["bboxes"]
        attention_mask = self.data_results[idx]["attention_mask"]
        token_type_id = self.data_results[idx]["token_type_ids"]
        label = self.data_results[idx]["ner_tags"]

        return {
            "input_ids": torch.tensor(input_id,dtype=torch.int64).flatten(),
            "bboxes":torch.tensor(bbox,dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask,dtype=torch.int64).flatten(),
            "token_type_ids":torch.tensor(token_type_id,dtype=torch.int64).flatten(),
            "ner_tags": torch.tensor(label,dtype=torch.int64).flatten()
        }