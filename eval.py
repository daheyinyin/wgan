""" test WGAN """
import os
import json
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
import numpy as np
from PIL import Image

from src.dcgan_model import DcganG
from src.dcgannobn_model import DcgannobnG
from src.args import get_args


if __name__ == "__main__":

    args_opt = get_args('eval')
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(device_id=args_opt.device_id)

    with open(args_opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())

    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    n_extra_layers = generator_config["n_extra_layers"]

    # generator
    if noBN:
        netG = DcgannobnG(imageSize, nz, nc, ngf, n_extra_layers)
    else:
        netG = DcganG(imageSize, nz, nc, ngf, n_extra_layers)

    # load weights
    load_param_into_net(netG, load_checkpoint(args_opt.ckpt_file))

    # initialize noise
    randgen = np.random.RandomState(args_opt.input_seed)
    input_array = randgen.normal(size=[args_opt.nimages, nz, 1, 1])
    fixed_noise = Tensor(input_array, dtype=mstype.float32)

    fake = netG(fixed_noise)
    mul = ops.Mul()
    add = ops.Add()
    reshape = ops.Reshape()
    fake = mul(fake, 0.5*255)
    fake = add(fake, 0.5*255)
    if args_opt.output_dir is None:
        args_opt.output_dir = './output'
    os.system('mkdir {0}'.format(args_opt.output_dir))
    for i in range(args_opt.nimages):
        img_pil = reshape(fake[i, ...], (1, nc, imageSize, imageSize))
        img_pil = img_pil.asnumpy()[0].astype(np.uint8).transpose((1, 2, 0))
        img_pil = Image.fromarray(img_pil)
        img_pil.save(os.path.join(args_opt.output_dir, "generated_%02d.png" % i))

    print("Generate images success!")
