import torch
import os

# this part is hard-coded and should be modified if changed dataset or model
expert_num = 3
prompt_len = 5
tokenizer_len = 50265
tokenizer_pad_token_id = 1
num_labels = 11
use_fair = False

if os.path.exists("pretrained_model/expert_prompt.bin"):
    expert_prompt = torch.load("pretrained_model/expert_prompt.bin")
if not os.path.exists("pretrained_model/expert_prompt.bin") or expert_prompt.shape != (expert_num, prompt_len):
    expert_prompt = torch.randint(low=1, high=50265, size=(expert_num, prompt_len))
    torch.save(expert_prompt, "pretrained_model/expert_prompt.bin")
device = torch.device("cuda")
expert_prompt = expert_prompt.to(device)


from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


def repeat(tensor, K):
    # [B, ...] => [B*K, ...] Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        expand_size = B, K, *size
        tensor = tensor.unsqueeze(1).expand(*expand_size).contiguous().view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out


class RobertaForTokenClassification_moe(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.prompt_len=prompt_len

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.is_train = True
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
        label_mask=None
    ):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if type(label_mask) != type(None):
            inputs["label_mask"] = label_mask
        moe_inputs = self.MoEDataCollator(inputs)
        return self.forward_inner(**moe_inputs)


    def forward_inner(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None
    ):

        outputs = self.roberta(
            input_ids[:, prompt_len:],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        final_embedding = outputs[0]
        
        # not necessary
        expert_inputs = input_ids[:, :prompt_len]
        expert_embedding = self.roberta(expert_inputs)[0][:, -1, :]
        expert_embedding = repeat(expert_embedding, final_embedding.shape[1]).view(final_embedding.shape[0], final_embedding.shape[1], -1)
        final_embedding = torch.cat([expert_embedding, final_embedding], dim=-1)
        
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)
        
        if not self.is_train:    # for evaluation model
            final_embedding = torch.stack([torch.mean(final_embedding[i:i+expert_num], dim=0) for i in range(0, len(final_embedding), expert_num)], dim=0)
            logits = torch.stack([torch.mean(logits[i:i+expert_num], dim=0) for i in range(0, len(logits), expert_num)], dim=0)
            attention_mask = torch.stack([attention_mask[i] for i in range(0, len(attention_mask), expert_num)], dim=0)
            if type(labels) != type(None):
                labels = torch.stack([labels[i] for i in range(0, len(labels), expert_num)], dim=0)
            self.is_train = True

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.reshape(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]

            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)


    def fair_expert_selection(self, loss):
        '''
            add the fairness code there, the loss is of the shape [batch_size, expert_num, 1] without softmax
        '''
        lamda = 0.1
        
        # score = torch.exp(-loss / lamda)
        score = loss
        
        score = score.squeeze()
        batch_size, expert_num = score.size()
        mu1 = torch.ones(batch_size, 1).to(device) / batch_size
        mu2 = torch.ones(expert_num, 1).to(device) / expert_num
        b = torch.ones(expert_num, 1).to(device) / expert_num
        for _ in range(30):
            a = mu1 / (score @ b)
            b = mu2 / (score.T @ a)
        score = torch.diag(a.squeeze()) @ score @ torch.diag(b.squeeze())
        score = torch.reshape(score, (batch_size, expert_num, 1))
        return score
    
    def MoEDataCollator(self, batch_inputs):
        batch_size = batch_inputs['input_ids'].shape[0]

        # construct prompt concatenated inputs
        mixture_ids_prompt = expert_prompt.repeat(batch_size, 1)
        mixture_inputs = {k: repeat(v, expert_num) for k, v in batch_inputs.items()}
        mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)

        if self.is_train:
            # choose experts by model best output (only for train)
            self.eval()
            _inputs = mixture_inputs.copy()
            _inputs = {k: v.to(device) for k, v in _inputs.items()}
            labels = _inputs.pop("labels")
            outputs = self.forward_inner(**_inputs)
            logits = outputs[0]

            loss_fct = KLDivLoss(reduction="mean") if logits.shape == labels.shape else CrossEntropyLoss(reduction="mean")
            loss = []
            for i in range(batch_size * expert_num):
                masks = _inputs["attention_mask"][i]
                if "label_mask" in _inputs:
                    masks = masks & _inputs["label_mask"][i].long()
                active_logits, active_labels = logits[i][masks == 1], labels[i][masks == 1]
                loss.append(loss_fct(active_logits, active_labels))
            loss = torch.stack(loss, dim=0).reshape(batch_size, expert_num, 1)
            if use_fair:
                loss = self.fair_expert_selection(loss)
            mixture_ids = loss.argmin(dim=1).type(torch.int64)

            batch_inputs_new = batch_inputs
            expanded_mixture_ids = mixture_ids.expand(batch_size, prompt_len).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(batch_size, expert_num, -1), dim=1, index=expanded_mixture_ids).squeeze()
            batch_inputs_new['input_ids'] = torch.cat([input_ids_prompt, batch_inputs_new['input_ids']], dim=1)
        else:
            batch_inputs_new = mixture_inputs

        return batch_inputs_new