import argparse, os

def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)

    parser.add_argument('--n_trn_epsd', type=int, default=30000)
    parser.add_argument('--n_tst_epsd', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--visdir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--csvfn', type=str, default=None)
    parser.add_argument('--cdim', type=int, default=64)
    parser.add_argument('--zdim', type=int, default=5)
    parser.add_argument('--hdim', type=int, default=512)

    parser.add_argument("--any", default=False, action="store_true")
    parser.add_argument("--trte", default=False, action="store_true")

    parser.add_argument('--n_qry_set', type=int, default=5)

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)

    parser.add_argument('--nclass', type=int, default=55)

    parser.add_argument('--sway', type=int, default=100)
    parser.add_argument('--bsize', type=int, default=256)

    parser.add_argument('--trclass', type=int, default=4800)
    parser.add_argument('--sptclass', type=int, default=200)

    parser.add_argument('--ratio_setdisc', type=int, default=4)

    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--prior', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument("--recon_mode", default=False, action="store_true")
    parser.add_argument("--set_exp", default=False, action="store_true")
    parser.add_argument("--vis_train", default=False, action="store_true")

    parser.add_argument("--vis_with_tedata", default=False, action="store_true")
    parser.add_argument("--vis_with_trtedata", default=False, action="store_true")
    parser.add_argument("--manifold", default=False, action="store_true")
    parser.add_argument("--reconstruct", default=False, action="store_true")
    parser.add_argument("--scatter", default=False, action="store_true")
    parser.add_argument("--scatter_full", default=False, action="store_true")
    parser.add_argument("--scatter_qzx", default=False, action="store_true")

    return parser.parse_args(argv)

def setup(args):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    os.environ['CUDA_CACHE_PATH'] = '/st1/dhna/tmp'

    savedir = '../' + args.model + '/results/base/sample_run' \
            if args.savedir is None else args.savedir
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    print ("Save dir  ", savedir)
    if args.vis_with_tedata == True and args.vis_with_trtedata == True:
        print ("args.vis_with_tedata and args.vis_with_trtedata both can't be True at the same time.")
        raise NotImplementedError()

    return savedir
