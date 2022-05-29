class Config:
    def __init__(self):
        self.data_dir = 'cmrc2018'
        self.output_dir = 'output'
        self.saved_model = "albert-base.bin"
        self.do_lower_case = True
        self.max_seq_len = 512
        self.train_batch_size = 32
        self.dev_batch_size = 32
        self.test_batch_size = 32
        self.num_train_epochs = 10
        self.data_cache = True

        # self.bert_model = './model_hub/voidful-albert-chinese-tiny/'
        self.bert_model = 'voidful/albert_chinese_base'
        self.device = None
        self.learning_rate = 3e-5
        self.clip_grad = 1
        self.checkpoint = 50
        self.export_model = True
        self.hidden_size = 312  # 312 for albert
        self.weight_start = 1
        self.weight_end = 1
        self.warmup_steps_ratio = 0.1
        self.dropout_prob = 0.1
        self.use_ori_albert = True