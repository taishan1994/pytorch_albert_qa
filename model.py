#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Bert Model for MRC-Based NER Task

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertConfig


class BertQA(nn.Module):
    def __init__(self, config):
        super(BertQA, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model, add_pooling_layer=False,
                 hidden_dropout_prob=config.dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            start_logits = torch.argmax(start_logits, 1)
            end_logits = torch.argmax(end_logits, 1)
            return start_logits, end_logits


class BertQA2(nn.Module):
    def __init__(self, config):
        super(BertQA2, self).__init__()
        # bert_config = BertConfig.from_pretrained(config.bert_model)
        # self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        self.hidden_size = config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.loss_wb = config.weight_start
        self.loss_we = config.weight_end

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
        """

        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)

        sequence_heatmap = sequence_output["last_hidden_state"]  # batch x seq_len x hidden
        start_logits = self.start_outputs(sequence_heatmap)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_heatmap)  # batch x seq_len x 2

        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss
            return total_loss
        else:
            start_logits = torch.argmax(start_logits, dim=-1)
            end_logits = torch.argmax(end_logits, dim=-1)

            return start_logits, end_logits
