import torch
import glob
import argparse
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from misc.data_loader import StockDataset, DataType

# 配置日志
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"train_{current_datetime}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler(f"data/logs/{log_filename}"),  # 写入日志文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

def gather_csv(train_path, valid_path):
    train_files = glob.glob(train_path)
    valid_files = glob.glob(valid_path)
    return train_files, valid_files

def load_data(args):
    """
    加载数据集并返回DataLoader
    """
    train_files, valid_files = gather_csv(args.train_path, args.valid_path)
    train_dataset = StockDataset(
        files_list=train_files,
        maxseq_len=args.max_seq_len,
        random_split_ratio=args.random_split_ratio,
        use_time_feature=args.use_time_feature,
        data_type=DataType(args.data_type),
        cache_size=args.cache_size,
        random_mask_ratio=args.random_mask_ratio,
    )
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataset = StockDataset(
        files_list=valid_files,
        maxseq_len=args.max_seq_len,
        random_split_ratio=args.random_split_ratio,
        use_time_feature=args.use_time_feature,
        data_type=DataType(args.data_type),
        cache_size=args.cache_size,
        random_mask_ratio=args.random_mask_ratio,
    )
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    return train_data_loader, valid_data_loader

def train_model(model, train_data_loader, valid_data_loader, optimizer, num_epochs, device):
    """
    训练模型
    """
    writer = SummaryWriter()
    total_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, data in enumerate(train_data_loader):
            seq = data[0].to(device).float()
            seq_len = data[1].to(device).int()
            time_feat = None
            if len(data) > 2:
                time_feat = data[2].to(device).float()
            optimizer.zero_grad()
            predict_out = model(seq, time_feat)
            loss, channel_loss = model.pretrain_loss(seq, seq_len, predict_out)
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()
            if channel_loss is not None:
                channel_loss = list(map(lambda x: x.item(), channel_loss))
                loss_str_list = []
                for loss_name, closs in zip(StockDataset.ChannelNames, channel_loss):
                    loss_str_list.append(f"{loss_name}@{closs}")
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, TotalBatch {total_step}, Loss: {avg_loss}, Channel Loss: {loss_str_list}')
                for idx in range(len(channel_loss)):
                    writer.add_scalar('Loss/train_' + StockDataset.ChannelNames[idx], channel_loss[idx], total_step)
            else:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, TotalBatch {total_step}, Loss: {avg_loss}')
            writer.add_scalar('Loss/batch', avg_loss, total_step)
            total_step += 1
        model.eval()
        with torch.no_grad():  # 禁用梯度计算
            valid_loss = 0
            batch_num = 0
            for batch_idx, data in enumerate(valid_data_loader):
                seq = data[0].to(device).float()
                seq_len = data[1].to(device).int()
                time_feat = None
                if len(data) > 2:
                    time_feat = data[2].to(device).float()
                predict_out = model(seq, time_feat)
                val_loss, channel_loss = model.pretrain_loss(seq, seq_len, predict_out)
                valid_loss += val_loss.item()
                batch_num += 1
            # 计算平均验证损失
            valid_loss /= batch_num
            if channel_loss is not None:
                channel_loss = list(map(lambda x: x.item() / batch_num, channel_loss))
                for idx in range(len(channel_loss)):
                    writer.add_scalar('Loss/valid_' + StockDataset.ChannelNames[idx], channel_loss[idx], epoch)
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss}, Channel Loss: {channel_loss}')
            else:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss}')
            writer.add_scalar('Loss/valid', valid_loss, epoch)
        # 重置数据加载器
        valid_data_loader.dataset.reset()
        train_data_loader.dataset.reset()
    writer.close()
    # 保存模型权重
    torch.save(model.state_dict(), 'data/models/model_weights.pth')

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LLM-TST model")
    parser.add_argument('--train_path', type=str, required=True, help="train path")
    parser.add_argument('--valid_path', type=str, required=True, help="valid path")
    parser.add_argument('--max_seq_len', type=int, default=300, help="Maximum sequence length")
    parser.add_argument('--random_split_ratio', type=float, default=0.3, help="Random split ratio for the dataset")
    parser.add_argument('--use_time_feature', action='store_true', help="Use time feature in the dataset")
    parser.add_argument('--data_type', type=int, help="Type of the data")
    parser.add_argument('--cache_size', type=int, default=10, help="Cache size for the dataset")
    parser.add_argument('--random_mask_ratio', type=float, default=0, help="Random mask ratio for the dataset")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to train")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mpi'])
    parser.add_argument('--model_arch', type=str, default='512|8|12')
    parser.add_argument('--is_mix_channel', action='store_true')
    parser.add_argument('--price_channel_index', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 定义模型参数
    n_dim, n_layer, n_head = args.model_arch.split("|")
    params = ModelArgs(
        dim=int(n_dim),
        n_layers=int(n_layer),
        n_heads=int(n_head),
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        n_channel=len(StockDataset.ChannelNames),
        device=args.device,
        use_time_feature=True,
        time_feature_dim=5,
        stride=12,
        patch_len=12
    )
    if not args.is_mix_channel:
        from model.llm_tst import LLMTST, ModelArgs
        print("use all channel pretrain...")
        # 实例化模型
        model = LLMTST(params)
    else:
        from model.llm_mix_tst import LLMMixTST, ModelArgs
        print("use mix channel pretrain...")
        params.price_channel = args.price_channel_index
        # 实例化模型
        model = LLMMixTST(params)        
    model.to(args.device)  # 将模型迁移到GPU
    # 实例化数据加载器
    train_data_loader, valid_data_loader = load_data(args)
    # 定义优化器
    optimizer = model.configure_optimizers(0.01, 0.0005, (0.9, 0.999), args.device)
    # 训练模型
    train_model(model, train_data_loader, valid_data_loader, optimizer, args.num_epochs, args.device)