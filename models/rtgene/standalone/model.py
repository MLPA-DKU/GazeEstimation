import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import models.rtgene.standalone.TDDFA.mobilenet_v1 as mobilenet_v1
import models.rtgene.standalone.TDDFA.utils.ddfa as ddfa
import models.rtgene.standalone.TDDFA.utils.inference as inf


class EyePatchExtractor:

    def __init__(self, device=None):

        self.device = 'cuda:0' if device is None else device
        self.detection_size = (120, 120)

        self.pretrained_model = getattr(mobilenet_v1, 'mobilenet_1')(num_classes=62)
        self.initialize_pretrained_model('TDDFA/models/phase1_wpdc_vdc.pth.tar')
        self.face_landmark = dlib.shape_predictor('TDDFA/models/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()

        self.transform = transforms.Compose([ddfa.ToTensorGjz(), ddfa.NormalizeGjz(mean=127.5, std=128)])

        self.eye_indices = np.array([36, 39, 42, 45])
        self.eye_patches_size = (60, 36)
        self.margin_ratio = 1.0
        self.desire_ratio = float(self.eye_patches_size[1]) / float(self.eye_patches_size[0]) / 2.0

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
        eye_landmarks = facial_landmarks[self.eye_indices]

        eye_width_l = eye_landmarks[3][0] - eye_landmarks[2][0]
        eye_width_r = eye_landmarks[1][0] - eye_landmarks[0][0]
        eye_center_l = (eye_landmarks[2][0] + eye_width_l / 2, (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0)
        eye_center_r = (eye_landmarks[0][0] + eye_width_r / 2, (eye_landmarks[0][1] + eye_landmarks[1][1]) / 2.0)

        eye_margin_l, eye_margin_r = eye_width_l * self.margin_ratio, eye_width_r * self.margin_ratio

        bounding_box_eye_l = np.zeros(4, dtype=np.int)
        bounding_box_eye_l[0] = eye_landmarks[2][0] - eye_margin_l / 2.0
        bounding_box_eye_l[1] = eye_landmarks[3][0] + eye_margin_l / 2.0
        bounding_box_eye_l[2] = eye_center_l[1] - (eye_width_l + eye_margin_l) * self.desire_ratio
        bounding_box_eye_l[3] = eye_center_l[1] + (eye_width_l + eye_margin_l) * self.desire_ratio

        bounding_box_eye_r = np.zeros(4, dtype=np.int)
        bounding_box_eye_r[0] = eye_landmarks[0][0] - eye_margin_r / 2.0
        bounding_box_eye_r[1] = eye_landmarks[1][0] + eye_margin_r / 2.0
        bounding_box_eye_r[2] = eye_center_r[1] - (eye_width_r + eye_margin_r) * self.desire_ratio
        bounding_box_eye_r[3] = eye_center_r[1] + (eye_width_r + eye_margin_r) * self.desire_ratio

        eye_patch_l = image[bounding_box_eye_l[2]:bounding_box_eye_l[3], bounding_box_eye_l[0]:bounding_box_eye_l[1], :]
        eye_patch_r = image[bounding_box_eye_r[2]:bounding_box_eye_r[3], bounding_box_eye_r[0]:bounding_box_eye_r[1], :]
        eye_patch_l = cv2.resize(eye_patch_l, self.eye_patches_size, interpolation=cv2.INTER_CUBIC)
        eye_patch_r = cv2.resize(eye_patch_r, self.eye_patches_size, interpolation=cv2.INTER_CUBIC)

        return eye_patch_l, eye_patch_r

    def initialize_pretrained_model(self, path):

        model_dict = self.pretrained_model.state_dict()
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']

        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]

        self.pretrained_model.load_state_dict(model_dict)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()


class GazeEstimator(nn.Module):

    def __init__(self, pretrained=False, device=None):
        super(GazeEstimator, self).__init__()
        self.device = 'cuda:0' if device is None else device
        self.backbone_l = models.vgg16(pretrained=pretrained).to(self.device).features
        self.backbone_r = models.vgg16(pretrained=pretrained).to(self.device).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear_l = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.linear_r = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.linear_x = nn.Sequential(
            nn.Linear(2048, 512),
        )
        self.predictor = nn.Sequential(
            nn.BatchNorm2d(514),
            nn.ReLU(),
            nn.Linear(514, 256),
            nn.Linear(256, 2),
        )

    def forward(self, l, r, h):
        l = self.backbone_l(l)
        l = self.avgpool(l)
        l = torch.flatten(l, 1)
        l = self.linear_l(l)

        r = self.backbone_r(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.linear_r(r)

        x = torch.cat((l, r), dim=1)
        x = self.linear_x(x)

        x = torch.cat((x, h), dim=1)
        gaze_prediction = self.predictor(x)
        return gaze_prediction
