import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
import models.alignment.TDDFA.mobilenet as mobilenet
import models.alignment.TDDFA.utils.inference as inf


class FaceAlignment:

    def __init__(self, device=None):
        self.device = 'cuda:0' if device is None else device
        self.detection_size = (120, 120)

        self.pretrained_model = getattr(mobilenet, 'mobilenet_1')(num_classes=62)
        self.initialize_pretrained_model('TDDFA/models/phase1_wpdc_vdc.pth.tar')
        self.face_landmark = dlib.shape_predictor('TDDFA/models/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.498, std=0.5),
        ])

        # self.eye_indices = np.array([36, 39, 42, 45])

    def __call__(self, image):
        boxes = self.face_detector(image, 1)
        roi_box = None
        for box in boxes:
            pts = self.face_landmark(image, box).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = inf.parse_roi_box_from_landmark(pts)

        cropped_image = inf.crop_img(image, roi_box)
        cropped_image = cv2.resize(cropped_image, dsize=self.detection_size, interpolation=cv2.INTER_LINEAR)
        cropped_image = self.transform(cropped_image).unsqueeze(0)

        with torch.no_grad():
            cropped_image = cropped_image.to(self.device)
            param = self.pretrained_model(cropped_image)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        facial_landmarks = inf.predict_68pts(param, roi_box).T
        return facial_landmarks

    def initialize_pretrained_model(self, path):
        model_dict = self.pretrained_model.state_dict()
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']

        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]

        self.pretrained_model.load_state_dict(model_dict)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()


if __name__ == '__main__':
    from PIL import Image
    import gc
    import os
    import matplotlib.pyplot as plt

    path = '/workspace/datasets/Gaze360/imgs/rec_000/head/000088/'
    files = os.listdir(path)

    for i, pt in enumerate(files):
        try:
            image = np.array(Image.open(path + pt).convert('RGB'))
            align = FaceAlignment()
            res = align(image)
            res = res[:, :2]

            plt.imshow(image)
            plt.scatter(res[:, 0], res[:, 1])
            plt.axis('off')
            plt.show()
            plt.close()

            gc.collect()

        except:
            continue

    breakpoint()
