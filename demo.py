from path import Path

import mmu
from conf import Conf
from models.model import DummyModel
from post_processing import PostProcessor
from pre_processing import PreProcessor


def demo(img_path, exp_name):
    # type: (str, str) -> None
    """
    Quick demo of the complete pipeline on a test image.

    :param img_path: path of the image you want to test
    :param exp_name: name of the experiment you want to test
    """

    cnf = Conf(exp_name=exp_name)

    # init model and load weights of the best epoch
    model = DummyModel()
    model.eval()
    model.requires_grad(False)
    model = model.to(cnf.device)
    model.load_w(cnf.exp_log_path / 'best.pth')

    # init pre- and post-processors
    pre_proc = PreProcessor(unsqueeze=True, device=cnf.device)
    post_proc = PostProcessor(out_ch_order='RGB')

    # read image and apply pre-processing
    img = mmu.imread(img_path)
    x = pre_proc.apply(img)

    # forward pass and post-processing
    y_pred = model.forward(x)
    img_pred = post_proc.apply(y_pred)

    # show input and output
    mmu.imshow(img, header='input', waitkey=0)
    mmu.imshow(img_pred, header='output')


if __name__ == '__main__':
    __p = Path(__file__).parent / 'dataset' / 'samples' / 'test' / '10_x.png'
    demo(img_path=__p, exp_name='default')
