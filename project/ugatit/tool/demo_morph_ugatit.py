import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.s_ugatit_plus import S_UGATITPlus
from configs.cfgs_s_ugatit_plus import test_cfgs as cfgs
import torch
import cv2
from PIL import Image
from importlib import import_module
from utils.utils import load_image, check_dir_existing
import numpy as np
from glob import glob
import argparse


def preprocessing(img):
    img = torch.from_numpy(img).float() / 255.0
    img = (img - 0.5) * 2
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

torch.set_grad_enabled(False)
face_align_lib = import_module('3rdparty.face_alignment.api')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resume', type=str, required=True)
    # parser.add_argument('--input', type=str, required=True)
    # parser.add_argument('--saved-dir', required=True, type=str)
    # parser.add_argument('--align', action='store_true', default=False)

    parser.add_argument('--resume', type=str, default='F:/WCY-code-all/wcy3/Morph-UGATIT-self/ckpt/100.pt')
    parser.add_argument('--input', type=str, default='F:/WCY-code-all/wcy3/Morph-UGATIT-self/dataset/skull2skin/testA')
    parser.add_argument('--saved-dir', default='F:/WCY-code-all/wcy3/Morph-UGATIT-self/result1', type=str)
    parser.add_argument('--align', action='store_true', default=False)
    args = parser.parse_args()


    model_type = 'morph_ugatit'  # 'ugatit_plus'
    images_path = args.input  # '/home/yangjie08/human_face'
    weight_loc = args.resume  # '/home/yangjie08/My-CycleGAN/ckpt_ugatit_plus/99.pt'
    saved_dir = args.saved_dir

    saved_dir = os.path.join(saved_dir, model_type)
    check_dir_existing(saved_dir)

    alignment_loc =  'F:/WCY-code-all/wcy3/Morph-UGATIT-self/face_landmark/shape_predictor_68_face_landmarks.dat'
    size = 256
    # repeat_time = 3

    images_set = glob(images_path + '/*')
    FAtool = face_align_lib.FaceAlignment(alignment_loc)

    model = S_UGATITPlus(cfgs)
    G_A = model.G
    weight_set = torch.load(weight_loc, map_location='cpu')

    G_A.load_state_dict(weight_set['G'])
    G_A.eval()
    if torch.cuda.is_available():
        G_A.cuda()
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    for image_path in images_set:
        image = load_image(image_path)  # RGB uint8
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if args.align:
            image = FAtool.rotate_align_crop(image, size)
        else:
            image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        faces = [image]
        if faces is None:
            continue
        human_name = os.path.splitext(os.path.split(image_path)[1])[0]
        print('starting to transform {}'.format(human_name))
        for idx, face in enumerate(faces):
            img_tensor = preprocessing(face).to(dev)
            repeat_set = [face]

            # for _ in range(repeat_time):
            fakeB = G_A.test_forward(img_tensor, 'AtoB')
            fakeB = (fakeB * 0.5) + 0.5
            fakeB = fakeB.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            fakeB = fakeB.astype(np.uint8)  # RGB
            repeat_set.append(fakeB)

            fig = np.zeros((256, 256*2, 3), dtype=np.uint8)
            fig1 = np.zeros((256, 256*2, 3), dtype=np.uint8)
            fig_idx = 0
            row = 0
            row1 = 1
            # for row in range(2):
            for col in range(2):
                fig[256*row:256*(row+1), 256*col:256*(col+1)] = repeat_set[fig_idx]
                fig_idx += 1
            #fig = Image.fromarray(fig).convert('L')
            fig = Image.fromarray(fig)
            fig.save(os.path.join(saved_dir, '{}_{}.jpg'.format(human_name, idx)))


            #fig1 = Image.fromarray(fakeB).convert('L')
            fig1 = Image.fromarray(fakeB)
            fig1.save(os.path.join(saved_dir, 'fakeB_{}_{}.jpg'.format(human_name, idx)))
    print('pred done, saving images to "{}" dir.'.format(saved_dir))
