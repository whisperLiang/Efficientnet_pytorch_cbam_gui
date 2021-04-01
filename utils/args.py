import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='underwater classify')

parser.add_argument('--batch_size', default=10, type=int, help='training batch size (default: 10)')
parser.add_argument('--lr', default=7.5e-5, type=float, help='initial learning rate (default: 7.5e-5)')
parser.add_argument('--image_size', default=456, type=int, help='input image size (default: 456)')
parser.add_argument('--epochs', default=100, type=int, help='epochs of training (default: 100)')
parser.add_argument('--snap_num', default=5, type=int, help='number of snapshot (default: 5)')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay value of optimizer (default: 0)')
parser.add_argument('--resize_scale', default=0.8, type=float, help='value of scale of random resized crop (default: 0.8)')
parser.add_argument('--erasing_prob', default=0.2, type=float, help='prob of random erasing (default: 0.2)')
parser.add_argument("--cutmix", default='True', action='store_true', help='using cutmix (default: True)')
parser.add_argument("--label_smooth", default='True', action='store_true', help='using label smooth (default: True)')
parser.add_argument('--model_path', default=None, type=str, help='model path for resume training')
parser.add_argument('--num_class', default=20, type=int, help='the num of calsses')
parser.add_argument("--log_path", default="log/", type=str, help="log path for storing the log")

train_args = parser.parse_args()
