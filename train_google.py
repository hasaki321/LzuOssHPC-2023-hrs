import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import load_config, get_model, get_dataloader, get_logger, dump_data
import logging

def train_step(batch, model, loss_fn, optimizer):
    global config
    images, target = batch
    images = images.to(config.dev)
    target = target.to(config.dev)

    optimizer.zero_grad()

    output, aux_logits2, aux_logits1 = model(images)
    loss0 = loss_fn(output, target)
    loss1 = loss_fn(aux_logits1, target)
    loss2 = loss_fn(aux_logits2, target)
    loss = loss0 + loss1 * 0.3 + loss2 * 0.3

    loss.backward()
    optimizer.step()

    acc = (output.argmax(dim=-1) == target).float().mean()

    return loss.item(), acc.item()


def train_epoch(train_loader, model, loss_fn, optimizer, epoch):
    global config
    show_bar = tqdm(train_loader,leave=False)
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


def train_google(logger):
    global config

    total_loss = []
    total_acc = []
    model = get_model(config)
    model.train()
    model.to(config.dev)
    train_loader, test_loader = get_dataloader(config.batch_size)

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

        if (epoch+1)//config.save_epoch:
            torch.save(model.state_dict(), config.save_pth)
            logger.info(f"save model parameter at epoch {epoch+1}")
    dump_data(total_loss,total_acc,config)



if __name__ == '__main__':
    config = load_config("./config/google.yml")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    train_google(logger)