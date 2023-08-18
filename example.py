import os
from tiktokx.train import Trainer, parse_args

if __name__ == '__main__':

    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    data_config = {
        'n_users': 12345, 
        'n_items': 67890
    }
    
    trainer = Trainer(data_config)
    
    best_recall, run_time = trainer.train()
    
    print(f'Best Recall@{args.Ks}: {best_recall}')
    print(f'Total Running Time: {run_time}')