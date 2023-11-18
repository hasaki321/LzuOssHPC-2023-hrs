import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import load_config, get_model, get_dataloader, dump_data


def train_step(batch, model, loss_fn, optimizer):
    global config
    images, target = batch
    images = images.to(config.dev)
    target = target.to(config.dev)

    output = model(images)
    loss = loss_fn(output, target)

    loss.backward()
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

    total_loss = [1,]
    total_acc = [1,]
    model = get_model(config)

    model.to(config.dev)
    train_loader, valid_loader, test_loader = get_dataloader(config.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    for epoch in range(config.num_epoch):
        loss_recoder, acc_recoder = train_epoch(train_loader, model, loss_fn, optimizer, epoch)
        logger.info(
            f"Epoch {epoch + 1}: loss={(sum(loss_recoder) / len(loss_recoder)):.5f} ,acc={(sum(acc_recoder) / len(acc_recoder)):.5f}")
        total_loss += loss_recoder
        total_acc += acc_recoder

        valid(valid_loader,model,loss_fn)

        if (epoch + 1) // config.save_epoch:
            torch.save(model.state_dict(), config.save_pth)
            logger.info(f"save model parameter at epoch {epoch + 1}")

    test(valid_loader,model,loss_fn)
    dump_data(total_loss, total_acc, config)


if __name__ == '__main__':
    config = load_config("./config/resnet.yml")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
