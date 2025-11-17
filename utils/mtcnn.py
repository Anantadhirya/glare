import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


def generate_bounding_box(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12
    reg = reg.permute(1, 0, 2, 3)
    mask = probs >= thresh
    mask_inds = mask.nonzero()
    if len(mask_inds) == 0:
        return torch.tensor([]), torch.tensor([])
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg_selected = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg_selected], dim=1)
    return boundingbox, image_inds


def nms(boxes, threshold, method='Union'):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i.item())
        if len(order) == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)
        inter = w * h
        if method == 'Min':
            ovr = inter / torch.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def batched_nms(boxes, scores, image_inds, threshold):
    from torchvision.ops.boxes import batched_nms as torch_batched_nms
    if len(boxes) == 0:
        return torch.tensor([])
    keep = torch_batched_nms(boxes, scores, image_inds, threshold)
    return keep


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = reg.reshape(reg.shape[2], reg.shape[3])
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).T
    return boundingbox


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = torch.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).T
    return bboxA


def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]
    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h
    return y, ey, x, ex


def extract_face(img, box, image_size=160, margin=0):
    margin_x = margin * (box[2] - box[0]) / (image_size - margin)
    margin_y = margin * (box[3] - box[1]) / (image_size - margin)
    raw_w, raw_h = img.size
    box_with_margin = [
        int(max(box[0] - margin_x / 2, 0)),
        int(max(box[1] - margin_y / 2, 0)),
        int(min(box[2] + margin_x / 2, raw_w)),
        int(min(box[3] + margin_y / 2, raw_h)),
    ]
    face = img.crop(box_with_margin).resize((image_size, image_size), Image.BILINEAR)
    face = np.array(face, dtype=np.float32)
    face = np.transpose(face, (2, 0, 1))
    return face


class MTCNN:
    def __init__(self, image_size=160, margin=0, min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709):
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pnet = PNetTorch().to(self.device).eval()
        self.rnet = RNetTorch().to(self.device).eval()
        self.onet = ONetTorch().to(self.device).eval()
    
    def load_weights(self, pnet_path, rnet_path, onet_path):
        self.pnet.load_state_dict(torch.load(pnet_path, map_location=self.device))
        self.rnet.load_state_dict(torch.load(rnet_path, map_location=self.device))
        self.onet.load_state_dict(torch.load(onet_path, map_location=self.device))
    
    def _detect_face(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        if len(img.shape) == 3:
            img = img[np.newaxis, :]
        
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float().to(self.device)
        
        batch_size = len(img)
        h, w = img.shape[2:4]
        m = 12.0 / self.min_face_size
        minl = min(h, w) * m
        
        scales = []
        scale = m
        while minl >= 12:
            scales.append(scale)
            scale = scale * self.factor
            minl = minl * self.factor
        
        all_boxes = []
        all_image_inds = []
        
        for scale in scales:
            hs = int(h * scale + 1)
            ws = int(w * scale + 1)
            im_data = F.interpolate(img, size=(hs, ws), mode='area')
            im_data = (im_data - 127.5) * 0.0078125
            
            with torch.no_grad():
                reg, probs = self.pnet(im_data)
            boxes, image_inds = generate_bounding_box(reg, probs[:, 1], scale, self.thresholds[0])
            
            if len(boxes) > 0:
                keep = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.5)
                boxes = boxes[keep]
                image_inds = image_inds[keep]
                all_boxes.append(boxes)
                all_image_inds.append(image_inds)
        
        if len(all_boxes) == 0:
            return [np.array([])], [np.array([])]
        
        boxes = torch.cat(all_boxes, dim=0)
        image_inds = torch.cat(all_image_inds, dim=0)
        
        keep = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes = boxes[keep]
        image_inds = image_inds[keep]
        
        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).T
        boxes = rerec(boxes)
        
        y, ey, x, ex = pad(boxes, w, h)
        
        if len(boxes) > 0:
            im_data_list = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[image_inds[k], :, (y[k]-1):ey[k], (x[k]-1):ex[k]].unsqueeze(0)
                    im_data_list.append(F.interpolate(img_k, size=(24, 24), mode='area'))
            
            if len(im_data_list) == 0:
                return [np.array([])], [np.array([])]
            
            im_data = torch.cat(im_data_list, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            
            with torch.no_grad():
                out = self._batch_forward(im_data, self.rnet)
            out0 = out[0].T
            out1 = out[1].T
            score = out1[1, :]
            ipass = score > self.thresholds[1]
            
            boxes = torch.cat([boxes[ipass, :4], score[ipass].unsqueeze(1)], dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].T
            
            if len(boxes) > 0:
                keep = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
                boxes = boxes[keep]
                image_inds = image_inds[keep]
                mv = mv[keep]
                boxes = bbreg(boxes, mv)
                boxes = rerec(boxes)
        
        points = torch.zeros(0, 5, 2, device=self.device)
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h)
            im_data_list = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[image_inds[k], :, (y[k]-1):ey[k], (x[k]-1):ex[k]].unsqueeze(0)
                    im_data_list.append(F.interpolate(img_k, size=(48, 48), mode='area'))
            
            if len(im_data_list) == 0:
                return [np.array([])], [np.array([])]
            
            im_data = torch.cat(im_data_list, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            
            with torch.no_grad():
                out = self._batch_forward(im_data, self.onet)
            out0 = out[0].T
            out1 = out[1].T
            out2 = out[2].T
            score = out2[1, :]
            points = out1
            ipass = score > self.thresholds[2]
            
            points = points[:, ipass]
            boxes = torch.cat([boxes[ipass, :4], score[ipass].unsqueeze(1)], dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].T
            
            w_i = boxes[:, 2] - boxes[:, 0] + 1
            h_i = boxes[:, 3] - boxes[:, 1] + 1
            points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
            points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
            points = torch.stack([points_x, points_y]).permute(2, 1, 0)
            boxes = bbreg(boxes, mv)
            
            boxes_np = torch.cat([boxes, torch.zeros(len(boxes), 1, device=self.device)], dim=1)
            keep = nms(boxes_np, 0.7, method='Min')
            boxes = boxes[keep]
            image_inds = image_inds[keep]
            points = points[keep]
        
        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()
        image_inds = image_inds.cpu().numpy()
        
        batch_boxes = []
        batch_points = []
        for b_i in range(batch_size):
            b_i_inds = np.where(image_inds == b_i)[0]
            batch_boxes.append(boxes[b_i_inds])
            batch_points.append(points[b_i_inds])
        
        return batch_boxes, batch_points
    
    def _batch_forward(self, im_data, model, batch_size=512):
        outputs = []
        for i in range(0, len(im_data), batch_size):
            batch = im_data[i:i+batch_size]
            outputs.append(model(batch))
        
        return tuple(torch.cat([out[i] for out in outputs], dim=0) for i in range(len(outputs[0])))
    
    def detect(self, img, landmarks=False):
        batch_boxes, batch_points = self._detect_face(img)
        
        boxes_list = []
        probs_list = []
        points_list = []
        
        for box, point in zip(batch_boxes, batch_points):
            if len(box) == 0:
                boxes_list.append(None)
                probs_list.append([None])
                points_list.append(None)
            else:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes_list.append(box[:, :4])
                probs_list.append(box[:, 4])
                points_list.append(point)
        
        if not isinstance(img, (list, tuple)):
            boxes_list = boxes_list[0]
            probs_list = probs_list[0]
            points_list = points_list[0]
        
        if landmarks:
            return boxes_list, probs_list, points_list
        return boxes_list, probs_list
    
    def __call__(self, img):
        img_rgb = img.convert("RGB")
        batch_boxes, _ = self._detect_face(img_rgb)
        
        boxes = batch_boxes[0]
        if len(boxes) == 0:
            return None
        
        box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
        box = boxes[box_order][0]
        
        face = extract_face(img_rgb, box[:4], self.image_size, self.margin)
        face = (face - 127.5) / 128.0
        
        return face
