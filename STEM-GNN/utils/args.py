import argparse


def get_args_pretrain(default_params=None):
    parser = argparse.ArgumentParser('Pretrain')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    # Encoder Parameters
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_layers', '--layers', type=int, default=2)
    parser.add_argument('--activation', '--act', type=str, default="relu")
    parser.add_argument('--backbone', type=str, default="sage")
    parser.add_argument('--normalize', type=str, default="batch", choices=['none', 'batch', 'layer'])
    parser.add_argument('--dropout', type=float, default=0.15)

    # VQ Parameters
    parser.add_argument('--code_dim', type=int, default=768)
    parser.add_argument('--codebook_size', type=int, default=128)
    parser.add_argument('--codebook_head', type=int, default=4)
    parser.add_argument('--codebook_decay', type=float, default=0.8)
    parser.add_argument('--commit_weight', type=float, default=10)
    parser.add_argument('--ortho_reg_weight', type=float, default=1)
    parser.add_argument('--ortho_reg_max_codes', type=int, default=32)

    # MoE Parameters
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--moe_layers', type=str, default='none', choices=['none', 'all', 'last'])
    parser.add_argument('--moe_experts', '--K', type=int, default=3)
    parser.add_argument('--moe_tau', '--tau', type=float, default=1.0)
    parser.add_argument('--lamda_env', type=float, default=0.0)

    # Pretrain Dataset
    parser.add_argument('--pretrain_dataset', '--pt_data', type=str, default="all")
    parser.add_argument('--pretrain_run_id', '--pt_run_id', type=str, default="")
    parser.add_argument('--pretrain_epochs', '--pt_epochs', '--epochs', type=int, default=50)
    parser.add_argument('--pretrain_lr', '--pt_lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_weight_decay', '--pt_decay', '--decay', type=float, default=1e-5)
    parser.add_argument('--pretrain_batch_size', '--pt_batch', type=int, default=1024)
    parser.add_argument('--feat_p', type=float, default=0.2)
    parser.add_argument('--edge_p', type=float, default=0.2)
    parser.add_argument('--topo_recon_ratio', type=float, default=0.1)
    parser.add_argument('--feat_lambda', type=float, default=100)
    parser.add_argument('--topo_lambda', type=float, default=0.01)
    parser.add_argument('--topo_sem_lambda', type=float, default=100)
    parser.add_argument('--sem_lambda', type=float, default=1)
    parser.add_argument('--sem_encoder_decay', type=float, default=0.99)
    parser.add_argument('--use_schedular', type=bool, default=True)

    if default_params:
        parser.set_defaults(**default_params)

    args = parser.parse_args()
    return vars(args)


def get_args_finetune(default_params=None):
    parser = argparse.ArgumentParser('Finetune')

    # General Parameters
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument("--setting", type=str, default="standard", choices=["standard"])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    # Pre-train Parameters
    parser.add_argument("--pretrain_dataset", '--pt_data', type=str, default="all")
    parser.add_argument('--pretrain_task', '--pt_task', type=str, default='all')
    parser.add_argument("--pretrain_model_epoch", '--pt_epochs', type=int, default=25)
    parser.add_argument('--pretrain_seed', '--pt_seed', type=int, default=42)
    parser.add_argument('--pretrain_run_id', '--pt_run_id', type=str, default="")
    parser.add_argument("--pretrain_path", type=str, default="")

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", '--act', type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # VQ Parameters
    parser.add_argument("--code_dim", type=int, default=768)
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--codebook_head", type=int, default=4)
    parser.add_argument("--codebook_decay", type=float, default=0.8)
    parser.add_argument("--commit_weight", type=float, default=0.25)
    parser.add_argument("--ortho_reg_weight", type=float, default=1)
    parser.add_argument("--ortho_reg_max_codes", type=int, default=32)
    parser.add_argument(
        "--use_vq",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use vector quantization in the decoder (1: use [default], 0: bypass).",
    )

    # MoE Parameters
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--moe_layers', type=str, default='none', choices=['none', 'all', 'last'])
    parser.add_argument('--moe_experts', '--K', type=int, default=3)
    parser.add_argument('--moe_tau', '--tau', type=float, default=1.0)
    parser.add_argument('--lamda_env', type=float, default=0.0)

    # Fine-Tune Parameters
    parser.add_argument("--finetune_dataset", "--dataset", "--data", type=str, default="cora")
    parser.add_argument(
        "--freeze_vq", type=int, default=1, choices=[0, 1],
        help="Freeze VQ parameters during finetuning (1: freeze [default], 0: update)."
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--finetune_epochs", "--epochs", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--finetune_lr", "--lr", type=float, default=1e-3)
    parser.add_argument(
        "--finetune_seed",
        type=int,
        default=None,
        help="Run a single specified seed; when set, repeat is forced to 1.",
    )

    # Model Parameters
    parser.add_argument("--separate_decoder_for_each_head", type=bool, default=True)
    parser.add_argument(
        "--decoder_jac_coeff",
        type=float,
        default=0.0,
        help="Jacobian regularization strength for the linear decoder (Frobenius norm).",
    )

    if default_params:
        parser.set_defaults(**default_params)

    args = parser.parse_args()
    return vars(args)
