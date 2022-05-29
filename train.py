import json
import os
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForQuestionAnswering

from data_loader import CMRCDataLoader
from model import BertQA
from config import Config
from decode_utils import decode_qa, calculate_metric, get_p_r_f

def load_data(config):
    print("-*-" * 10)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, return_dict=False)
    dataset_loaders = CMRCDataLoader(config, tokenizer, mode="train", )
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train")
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    num_train_steps = dataset_loaders.get_num_train_epochs()
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps


def load_model(config, t_total):
    if config.use_ori_albert:
      model = AutoModelForQuestionAnswering.from_pretrained(config.bert_model)
    else:
      model = BertQA(config)
    model.to(config.device)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = config.warmup_steps_ratio * t_total,
                                                num_training_steps = t_total)

    return model, optimizer, scheduler


def train(config):
    train_dataloader, \
    dev_dataloader, text_dataloader, \
    num_train_steps = load_data(config)
    device = config.device
    model, optimizer, scheduler = load_model(config, num_train_steps)
    tr_loss = 0.0
    nb_tr_examples = 0
    nb_tr_steps = 0

    dev_best_precision = 0.0
    dev_best_recall = 0.0
    dev_best_f1 = 0.0
    dev_best_loss = float("inf")

    model.train()
    for idx in range(int(config.num_train_epochs)):
        print("#######" * 10)
        print("EPOCH: ", str(idx + 1))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_pos, end_pos = batch
            if config.use_ori_albert:
              output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
              loss = output.loss
            else:
              loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                          start_positions=start_pos, end_positions=end_pos)
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if nb_tr_steps > 0 and nb_tr_steps % config.checkpoint == 0:
                print("-*-" * 15)
                # print("current training loss is : ")
                # print(loss.item())
                print("【train】 epoch:{}/{} step:{}/{} loss:{}".format(
                    idx+1, config.num_train_epochs, nb_tr_steps, num_train_steps, loss.item()
                ))
                tmp_dev_loss, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval(model,
                                                      dev_dataloader,
                                                      config,
                                                      )
                print("......" * 10)
                print("DEV: loss, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1:
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    # export model
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir,
                                                         config.saved_model)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("SAVED model path is :")
                        print(output_model_file)

                    # tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    # print("......"*10)
                    # print("TEST: loss, acc, precision, recall, f1")
                    # print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    # test_acc_when_dev_best = tmp_test_acc
                    # test_pre_when_dev_best = tmp_test_prec
                    # test_rec_when_dev_best = tmp_test_rec
                    # test_f1_when_dev_best = tmp_test_f1
                    # test_loss_when_dev_best = tmp_test_loss

                    print("-*-" * 15)

                    print("=&=" * 15)
                    print("Best DEV : overall best loss, precision, recall, f1 ")
                    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1)
                    # print("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
                    # print(test_loss_when_dev_best, test_acc_when_dev_best, test_pre_when_dev_best, test_rec_when_dev_best, test_f1_when_dev_best)
                    print("=&=" * 15)
    print("Best DEV : overall best loss, precision, recall, f1 ")
    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1)



def eval(model, eval_dataloader, config):
    device = config.device
    model.eval()
    eval_loss = 0

    eval_steps = 0
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for input_ids, input_mask, segment_ids, start_pos, end_pos in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)

        with torch.no_grad():
            if config.use_ori_albert:
              output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
              tmp_eval_loss = output.loss
              start_logits = output.start_logits
              end_logits = output.end_logits
              start_logits = torch.argmax(start_logits, 1)
              end_logits = torch.argmax(end_logits, 1)
            else:
              tmp_eval_loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos)
              start_logits, end_logits = model(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy()
        end_label = end_logits.detach().cpu().numpy()
        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        eval_loss += tmp_eval_loss.mean().item()
        # mask_lst += input_mask
        eval_steps += 1

        # start_pred_lst += start_label
        # end_pred_lst += end_label
        #
        # start_gold_lst += start_pos
        # end_gold_lst += end_pos


        pred_res = decode_qa(start_label, end_label)
        true_res = decode_qa(start_pos, end_pos)
        tp, fp, fn = calculate_metric(true_res, pred_res)
        tp_all += tp
        fp_all += fp
        fn_all += fn

    eval_precision, eval_recall, eval_f1 = get_p_r_f(tp_all, fp_all, fn_all)
    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)

    return average_loss, eval_precision, eval_recall, eval_f1



def predict(model, eval_dataloader, config, tokenizer):
  device = config.device
  model.eval()
  for input_ids, input_mask, segment_ids, start_pos, end_pos in eval_dataloader:
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = segment_ids.to(device)
      start_pos = start_pos.to(device)
      end_pos = end_pos.to(device)

      with torch.no_grad():
          if config.use_ori_albert:
            output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            start_logits = output.start_logits
            end_logits = output.end_logits
            start_logits = torch.argmax(start_logits, 1)
            end_logits = torch.argmax(end_logits, 1)
          else:
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)
      start_logits = start_logits.detach().cpu().numpy().tolist()
      end_logits = end_logits.detach().cpu().numpy().tolist()

      start_pos = start_pos.to("cpu").numpy().tolist()
      end_pos = end_pos.to("cpu").numpy().tolist()
 
      for start, end, true_start, true_end, input_id in zip(start_logits, end_logits, start_pos, end_pos, input_ids):

        pred_anser_id = input_id[start:end+1]
        pred_answer = tokenizer.decode(pred_anser_id)
        true_answer_id = input_id[true_start:true_end+1]
        true_answer = tokenizer.decode(true_answer_id)
        # answer = tokenizer.
        print(pred_answer)
        print(true_answer)
        print("=" * 100)
      break

if __name__ == '__main__':
    args_config = Config()
    args_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_train = True
    do_test = False
    do_predict = False
    if do_train:
      train(args_config)
    """
    albert-tiny:
      DEV: loss, precision, recall, f1
      1.4955 0.334 0.3401 0.337
    albert-small:
      Best DEV : overall best loss, precision, recall, f1 
      1.242 0.419 0.4266 0.4228
    albert-base:2个epoch
      DEV: loss, precision, recall, f1 
      0.9623 0.5043 0.5278 0.5158
    """


    if do_test:
      if args_config.use_ori_albert:
        model = AutoModelForQuestionAnswering.from_pretrained(args_config.bert_model)
      else:
        model = BertQA(args_config)

      checkpoint = torch.load(os.path.join(args_config.output_dir, args_config.saved_model))
      tokenizer = BertTokenizer.from_pretrained(args_config.bert_model)
      model.load_state_dict(checkpoint)
      model.to(args_config.device)
      train_dataloader, \
      dev_dataloader, text_dataloader, \
      num_train_steps = load_data(args_config)
      average_loss, eval_precision, eval_recall, eval_f1 = eval(model, dev_dataloader, args_config)
      print("Best DEV : overall best loss, precision, recall, f1 ")
      print(average_loss, eval_precision, eval_recall, eval_f1)

    if do_predict:
      if args_config.use_ori_albert:
        model = AutoModelForQuestionAnswering.from_pretrained(args_config.bert_model)
      else:
        model = BertQA(args_config)

      checkpoint = torch.load(os.path.join(args_config.output_dir, args_config.saved_model))
      tokenizer = BertTokenizer.from_pretrained(args_config.bert_model)
      model.load_state_dict(checkpoint)
      model.to(args_config.device)
      train_dataloader, \
      dev_dataloader, text_dataloader, \
      num_train_steps = load_data(args_config)
      predict(model, dev_dataloader, args_config, tokenizer)

