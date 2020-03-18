import torch
import os
import numpy as np
from argparse import Namespace
import torch.optim as optim
from tqdm import tqdm_notebook
from torch.nn import functional as F
# my own module
from utils.batch import *
from nmt.model import NMTModel
from utils.dataset import NMTDataset


class Trainer:
    """The training routine"""
    
    def __init__(self, args):
        self.seed = args.seed
        self.cuda = args.cuda
        self.set_device()
        # Set seed for reproducibility
        self.set_seed_everywhere()
        # handle dirs
        self.handle_dirs(args.save_dir + "/exp" + str(args.exp_num))
        # set dataset and vectorizer
        self.set_dataset_and_vectorizer(args.tra_src, args.tra_tgt, 
                                   args.val_src, args.val_tgt, 
                                   args.tst_src, args.tst_tgt, 
                                   args.vectorizer_file)
        # set model
        self.set_model(source_vocab_size=len(self.vectorizer.source_vocab), 
                       source_embedding_size=args.source_embedding_size, 
                       target_vocab_size=len(self.vectorizer.target_vocab),
                       target_embedding_size=args.target_embedding_size, 
                       encoding_size=args.encoding_size,
                       decoding_size=args.decoding_size,
                       encoder_layers=args.encoder_layers,
                       decoder_layers=args.decoder_layers,
                       encoder_dropout=args.encoder_dropout,
                       decoder_dropout=args.decoder_dropout,
                       target_bos_index=self.vectorizer.target_vocab.begin_seq_index,
                       reload_from_files=args.reload_from_files,
                       model_state_file=args.model_state_file)
        
        # set for training
        self.set_optimizer(args.learning_rate)
        self.set_lr_scheduler(factor=0.75, patience=0)
        self.set_mask_index()
        self.set_train_state(args)
        self.args = args
        
        print('-----------------------------------------------')
        print('###### The parameters in this model are: #######')
        for name, param in self.model.named_parameters():
            print(name, tuple(param.size()))
        print('-----------------------------------------------')

        
    def set_dataset_and_vectorizer(self, 
                                   tra_src, tra_tgt, 
                                   val_src, val_tgt, 
                                   tst_src, tst_tgt, 
                                   vectorizer_file):
        
        self.dataset = NMTDataset.load_dataset_and_load_vectorizer(tra_src, tra_tgt, 
                                                                   val_src, val_tgt, 
                                                                   tst_src, args.tst_tgt, 
                                                                   vectorizer_file)
        self.vectorizer = self.dataset.get_vectorizer()
        
        
    def set_model(self, 
                 source_vocab_size, source_embedding_size, 
                 target_vocab_size, target_embedding_size, 
                 encoding_size, decoding_size,
                 encoder_layers, decoder_layers,
                 encoder_dropout, decoder_dropout,
                 target_bos_index,
                 # parameters for loading existing models
                 reload_from_files=False, model_state_file=None):
        
        if self.vectorizer: 
            self.model = NMTModel(source_vocab_size=source_vocab_size, source_embedding_size=source_embedding_size, 
                                 target_vocab_size=target_vocab_size, target_embedding_size=target_embedding_size, 
                                 encoding_size=encoding_size, decoding_size=decoding_size,
                                 encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                                 encoder_dropout=encoder_dropout, decoder_dropout = decoder_dropout,
                                 target_bos_index=target_bos_index)
        else:
            print("Warning! Please load the dataset first!")
        
        if reload_from_files and os.path.exists(model_state_file):
            self.model.load_state_dict(torch.load(model_state_file))
            print("Reloaded model")
        else:
            print("New model")
            
    def set_optimizer(self, learning_rate, type='Adam'):
        if type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            print("Unimplemented type of optimizer...")
        
    
    def set_lr_scheduler(self, factor=0.9, patience=1):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              mode='min', factor=factor,
                                                              patience=patience)
        
    def set_mask_index(self):
        self.mask_index = self.vectorizer.target_vocab.mask_index

    def set_train_state(self, args):
        self.train_state = self.make_train_state(args)
        
    def set_device(self):
        # Check CUDA
        if not torch.cuda.is_available():
            self.cuda = False

        self.device = torch.device("cuda" if self.cuda else "cpu")
    
        print("Using CUDA: {}".format(self.cuda))
    
    def set_seed_everywhere(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed_all(self.seed)
            
    @staticmethod
    def handle_dirs(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    @staticmethod
    def make_train_state(args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val_loss': 1e8,
                'early_stopping_best_val_acc': -1e8,
                'learning_rate': args.learning_rate,
                'lr': [],
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': args.model_state_file}

    @staticmethod
    def update_train_state(args, model, train_state, select='acc'):
        """Handle the training state updates.
        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better

        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

        # Save one model at least
        if train_state['epoch_index'] == 0:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]
            acc_tm1, acc_t = train_state['val_acc'][-2:]

            if select == 'loss':
                # If loss worsened
                if loss_t >= loss_tm1:
                    # Update step
                    train_state['early_stopping_step'] += 1
                # Loss decreased
                else:
                    # Save the best model
                    if loss_t < train_state['early_stopping_best_val_loss']:
                        torch.save(model.state_dict(), train_state['model_filename'])
                        train_state['early_stopping_best_val_loss'] = loss_t
                        train_state['early_stopping_best_val_acc'] = acc_t

                    # Reset early stopping step
                    train_state['early_stopping_step'] = 0
                    
            elif select == 'acc':
                # If acc worsened
                if acc_t <= acc_tm1:
                    # Update step
                    train_state['early_stopping_step'] += 1
                # acc increased
                else:
                    # Save the best model
                    if acc_t > train_state['early_stopping_best_val_acc']:
                        torch.save(model.state_dict(), train_state['model_filename'])
                        train_state['early_stopping_best_val_loss'] = loss_t
                        train_state['early_stopping_best_val_acc'] = acc_t

                    # Reset early stopping step
                    train_state['early_stopping_step'] = 0                    

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state

    @staticmethod
    def normalize_sizes(y_pred, y_true):
        """Normalize tensor sizes

        Args:
            y_pred (torch.Tensor): the output of the model
                If a 3-dimensional tensor, reshapes to a matrix
            y_true (torch.Tensor): the target predictions
                If a matrix, reshapes to be a vector
        """
        if len(y_pred.size()) == 3:
            y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
        if len(y_true.size()) == 2:
            y_true = y_true.contiguous().view(-1)
        return y_pred, y_true

    @classmethod
    def compute_accuracy(cls, y_pred, y_true, mask_index):
        
        y_pred, y_true = cls.normalize_sizes(y_pred, y_true)

        _, y_pred_indices = y_pred.max(dim=1)

        correct_indices = torch.eq(y_pred_indices, y_true).float()
        valid_indices = torch.ne(y_true, mask_index).float()

        n_correct = (correct_indices * valid_indices).sum().item()
        n_valid = valid_indices.sum().item()

        return n_correct / n_valid * 100

    @classmethod
    def sequence_loss(cls, y_pred, y_true, mask_index):
        y_pred, y_true = cls.normalize_sizes(y_pred, y_true)
        return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)
    
    
    def train(self):
        
        args = self.args
        
        # use tqdm_notebook to report status
        epoch_bar = tqdm_notebook(desc='training routine', 
                          total=args.num_epochs,
                          position=0)

        self.dataset.set_split('train')
        train_bar = tqdm_notebook(desc='split=train',
                                  total=self.dataset.get_num_batches(args.batch_size), 
                                  position=1, 
                                  leave=True)
        self.dataset.set_split('val')
        val_bar = tqdm_notebook(desc='split=val',
                                total=self.dataset.get_num_batches(batch_size=self.dataset.__len__()), 
                                position=1, 
                                leave=True)
        
        self.model = self.model.to(self.device)

        try:
            for epoch_index in range(args.num_epochs):
                sample_probability = (20 + epoch_index) / args.num_epochs

                self.train_state['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on
                self.dataset.set_split('train')
                batch_generator = generate_nmt_batches(self.dataset, 
                                                       batch_size=args.batch_size, 
                                                       device=self.device)
                running_loss = 0.0
                running_acc = 0.0
                self.model.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:

                    # --------------------------------------    
                    # step 1. zero the gradients
                    self.optimizer.zero_grad()

                    # step 2. compute the output
                    y_pred = self.model(batch_dict['x_source'], 
                                   batch_dict['x_source_length'], 
                                   batch_dict['x_target'],
                                   sample_probability=sample_probability)

                    # step 3. compute the loss
                    loss = self.sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)

                    # step 4. use loss to produce gradients
                    loss.backward()

                    # step 5. use optimizer to take gradient step
                    self.optimizer.step()

                    # -----------------------------------------
                    # compute the running loss and running accuracy
                    running_loss += (loss.item() - running_loss) / (batch_index + 1)
                    
                    acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)


                    for param_group in self.optimizer.param_groups:
                        _lr = param_group['lr']
                        break

                    # update bar
                    train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                          epoch=epoch_index, lr=_lr, 
                                          ppl=2 ** running_loss,
                                          patience=str(self.train_state['early_stopping_step']) \
                                                     + '/' \
                                                     + str(args.early_stopping_criteria))
                    train_bar.update()

                self.train_state['train_loss'].append(running_loss)
                self.train_state['train_acc'].append(running_acc)
                self.train_state['lr'].append(_lr)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                self.dataset.set_split('val')
                batch_generator = generate_nmt_batches(self.dataset, 
                                                       batch_size=self.dataset.__len__(), 
                                                       device=self.device)
                running_loss = 0.
                running_acc = 0.
                self.model.eval()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # compute the output
                    y_pred = self.model(batch_dict['x_source'], 
                                        batch_dict['x_source_length'], 
                                        batch_dict['x_target'],
                                        sample_probability=sample_probability)

                    # step 3. compute the loss
                    loss = self.sequence_loss(y_pred, batch_dict['y_target'], self.mask_index)

                    # compute the running loss and accuracy
                    running_loss += (loss.item() - running_loss) / (batch_index + 1)

                    acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # Update bar
                    val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                    val_bar.update()

                self.train_state['val_loss'].append(running_loss)
                self.train_state['val_acc'].append(running_acc)

                self.train_state = self.update_train_state(args=args, 
                                                           model=self.model, 
                                                           train_state=self.train_state,
                                                           select=args.model_select_criteria)

                self.scheduler.step(self.train_state['val_loss'][-1])

                if self.train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                
                # the best val acc is not necessarily the 'best', it is generated by the min-loss checkpoint
                epoch_bar.set_postfix(best_val_loss=self.train_state['early_stopping_best_val_loss'],
                                      best_val_acc=self.train_state['early_stopping_best_val_acc'])
                epoch_bar.update()

        except KeyboardInterrupt:
            print("Exiting loop")

args = Namespace(tra_src="data/en2ch.train.src",
                 tra_tgt="data/en2ch.train.tgt",
                 val_src="data/en2ch.valid.src",
                 val_tgt="data/en2ch.valid.tgt",
                 tst_src="data/en2ch.dev.src",
                 tst_tgt="data/en2ch.dev.tgt",
                 vectorizer_file="vocabs/vocab.EnCh.json",
                 model_state_file="experiments/exp2/model.pth",
                 exp_num=2,
                 save_dir="experiments",
                 reload_from_files=False,
                 expand_filepaths_to_save_dir=True,
                 cuda=True,
                 seed=1000,
                 learning_rate=0.003,
                 batch_size=50,
                 num_epochs=100,
                 early_stopping_criteria=3,
                 model_select_criteria='loss',
                 source_embedding_size=512, 
                 target_embedding_size=512,
                 encoding_size=512,
                 decoding_size=1024,
                 encoder_layers=2,
                 decoder_layers=1,
                 encoder_dropout=0.1,
                 decoder_dropout=0.1,
                 catch_keyboard_interrupt=True)

trainer = Trainer(args)
trainer.train()
