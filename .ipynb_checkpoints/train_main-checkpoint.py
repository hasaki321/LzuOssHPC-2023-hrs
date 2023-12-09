import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import *
from model import GoogLeNet
import time
import argparse
from torch.optim import lr_scheduler
from sam import SAM


def train_step(batch, model, loss_fn, optimizer):
    global config
    images, target = batch
    images = images.to(config.dev)
    target = target.to(config.dev)

    if  isinstance(model,GoogLeNet):
        output,out2,out3 = model(images)
        loss1 = loss_fn(output, target)
        loss2 = loss_fn(out2, target)
        loss3 = loss_fn(out3, target)
        loss = loss1*0.6 + loss2*0.2 + loss3*0.2
    else:
        if isinstance(model,EffNetV2):
            enable_running_stats(model)
        output = model(images)
        loss = loss_fn(output, target)

    loss.backward()

    if isinstance(model,EffNetV2):
        optimizer.first_step(zero_grad=True)
  
        disable_running_stats(model)
        loss_fn(model(images), target).backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.step()
        optimizer.zero_grad()


    acc = (output.argmax(dim=-1) == target).float().mean()

    return loss.item(), acc.item()


def train_epoch(train_loader, model, loss_fn, optimizer, epoch):
    global config
    model.train()
    show_bar = tqdm(train_loader, leave=False)
    show_bar.set_description(f'[Training Epoch: {epoch + 1}]')
    acc_recoder = []
    loss_recoder = []
    for idx, batch in enumerate(show_bar):
        loss, acc = train_step(batch, model, loss_fn, optimizer)
        loss_recoder.append(loss)
        acc_recoder.append(acc)

        if (idx + 1) % 5 == 0:
            show_bar.set_postfix({'loss': f'{loss:.5f}', 'acc': f'{acc:.4f}'})
    return loss_recoder, acc_recoder


def valid(valid_loader, model, loss_fn):
    show_bar = tqdm(valid_loader, leave=False)
    show_bar.set_description(f'[Valid]')
    model.eval()
    acc_recoder = []
    loss_recoder = []
    for idx, batch in enumerate(show_bar):
        images, target = batch
        images = images.to(config.dev)
        target = target.to(config.dev)

        output = model(images)

        loss = loss_fn(output, target).item()
        acc = (output.argmax(dim=-1) == target).float().mean().item()

        if (idx + 1) % 5 == 0:
            show_bar.set_postfix({'loss': f'{loss:.5f}', 'acc': f'{acc:.4f}'})
        loss_recoder.append(loss)
        acc_recoder.append(acc)
    logger.info(
        f'Valid loss={(sum(loss_recoder) / len(loss_recoder)):.5f} ,acc={(sum(acc_recoder) / len(acc_recoder)):.5f}')
    return loss_recoder,acc_recoder


def test(test_loader, model, loss_fn):
    show_bar = tqdm(test_loader, leave=False)
    show_bar.set_description(f'[Test]')
    model.eval()
    acc_recoder = []
    loss_recoder = []
    for idx, batch in enumerate(show_bar):
        images, target = batch
        images = images.to(config.dev)
        target = target.to(config.dev)

        output = model(images)

        loss = loss_fn(output, target).item()
        acc = (output.argmax(dim=-1) == target).float().mean().item()

        if (idx + 1) % 5 == 0:
            show_bar.set_postfix({'loss': f'{loss:.5f}', 'acc': f'{acc:.4f}'})
        loss_recoder.append(loss)
        acc_recoder.append(acc)
    logger.info(
        f'Test loss={(sum(loss_recoder) / len(loss_recoder)):.5f} ,acc={(sum(acc_recoder) / len(acc_recoder)):.5f}')


def main():
    global config

    logger.info(f"---model {config.model} start training---")
    start_time = time.time()
    total_loss = []
    total_acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 0
    model = get_model(config)

    model.to(config.dev)
    train_loader, valid_loader, test_loader = get_dataloader(config.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    if isinstance(model,EffNetV2):
        optimizer = SAM(model.parameters(), optim.Adam, lr=config.learning_rate)
    else:
        optimizer = optim.Adam(params=model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(1 - step / config.num_epoch, 0.2))

    for epoch in range(config.num_epoch):
        loss_recoder, acc_recoder = train_epoch(train_loader, model, loss_fn, optimizer, epoch)
        logger.info(
            f"Epoch {epoch + 1}: loss={(sum(loss_recoder) / len(loss_recoder)):.5f} ,acc={(sum(acc_recoder) / len(acc_recoder)):.5f}")
        total_loss += loss_recoder
        total_acc += acc_recoder

        vloss_recoder,vacc_recoder = valid(valid_loader, model, loss_fn)
        valid_loss += vloss_recoder
        valid_acc += vacc_recoder

        if (sum(vacc_recoder)/len(vacc_recoder))>best_acc:
            best_acc = sum(vacc_recoder)/len(vacc_recoder)
            torch.save(model.state_dict(), config.save_pth)
            logger.info(f"save model parameter at epoch {epoch + 1}")
        scheduler.step()

    end_time = time.time()
    train_time = end_time-start_time
    # training_time_formatted = time.strftime("%H:%M:%S", time.gmtime(train_time))

    test(test_loader, model, loss_fn)
    dump_data(config,total_loss, total_acc, valid_loss,valid_acc)

    logger.info(f"training model {config.model} finish")
    logger.info(f"time cost: {train_time}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('--model', type=str, help='模型类型')
    args = parser.parse_args()
    model = args.model
    config = load_config(f"./config/{model}.yml")
    logging.basicConfig(filename=f'save/train_{model}.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
