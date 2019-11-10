import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--freeze_id',
        default=True,
        type=bool,
        help='if freeze id-network in netG training'
    )
    parser.add_argument(
        '--num_segments',
        default=7,
        type=int,
        help='number of video segments'
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='batch size'
    )
    parser.add_argument(
        '--epochs',
        default=200,
        type=int,
        help='total epoches u want to train'
    )
    parser.add_argument(
        '--sstep',
        default=120,
        type=int,
        help='number of epoches of lr decay with rate of 0.1'
    )
    parser.add_argument(
        '--eval_freq',
        default=1,
        type=int,
        help='frequency of validation and save models'
    )
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='momentum'
    )
    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='input image size facing model'
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help='learning rate'
    )
    parser.add_argument(
        '--Alpha',
        default=5,
        type=int,
        help='in netG loss, Alpha * loss_exp - loss_id'
    )
    parser.add_argument(
        '--weight_decay',
        default=0.008,
        type=float,
        help='weight decay'
    )
    parser.add_argument(
        '--dataset',
        default='oulu',
        type=str,
        help='dataset u test on'
    )
    args = parser.parse_args()
    return args