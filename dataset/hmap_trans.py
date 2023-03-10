import cv2
import numpy as np


def generate_hmap(instances, image):
    img_h, img_w = image.height, image.width
    x_range = np.arange(0, img_w)
    y_range = np.arange(0, img_h)
    x_map, y_map = np.meshgrid(x_range, y_range)

    heatmap = np.zeros((img_h, img_w, 3), dtype=np.float32)

    for instance in instances:
        rrect = instance[:5]
        label_id = instance[5]
        x_center, y_center = rrect[0], rrect[1]
        rrect_w, rrect_h = rrect[2], rrect[3]
        rot_angle = rrect[4]

        # 1. Calculate the minium bounding box of the rotated rectangle
        box_points = cv2.boxPoints(((x_center, y_center), (rrect_w, rrect_h), rot_angle))
        box_x1, box_y1, box_w, box_h = cv2.boundingRect(box_points)
        box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h
        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(img_w, box_x2)
        box_y2 = min(img_h, box_y2)

        # 2. Calculate the weight of the box patch
        patch_w, patch_h = box_w//1.5,  box_h//1.5
        patch_x1, patch_y1 = box_x1 + patch_w//2, box_y1 + patch_h//2
        patch_x2, patch_y2 = box_x2 - patch_w//2, box_y2 - patch_h//2
        if (patch_x2 - patch_x1) < 1 or (patch_y2 - patch_y1) < 1:
            patch_weight = 0
        else:
            patch = image.crop((patch_x1, patch_y1, patch_x2, patch_y2))
            patch_weight = calc_patch_weight(patch)

        # 2. The line function of the box's axes (a*x+b*y+c)
        a1 = np.cos(np.deg2rad(rot_angle))
        b1 = np.sin(np.deg2rad(rot_angle))
        c1 = -a1 * x_center - b1 * y_center
        const1 = np.sqrt(a1 ** 2 + b1 ** 2)

        a2 = np.cos(np.deg2rad(rot_angle + 90))
        b2 = np.sin(np.deg2rad(rot_angle + 90))
        c2 = -a2 * x_center - b2 * y_center
        const2 = np.sqrt(a2 ** 2 + b2 ** 2)

        # 3. Calculate the distance of each pixel to the box's axes line to generate the gaussian heatmap
        # adjust the kernel size according to the patch weight
        kernel_w = rrect_w * patch_weight
        kernel_h = rrect_h * patch_weight
        sigma1 = 0.3 * ((kernel_w-1) * 0.5 - 1) + 0.8
        sigma2 = 0.3 * ((kernel_h-1) * 0.5 - 1) + 0.8
        x = x_map[box_y1:box_y2, box_x1:box_x2]
        y = y_map[box_y1:box_y2, box_x1:box_x2]
        d1 = np.abs(a1 * x + b1 * y + c1) / const1  # distance to the first axis
        g1 = np.exp(-d1 ** 2 / (2 * sigma1 ** 2))  # gaussian distribution along the first axis
        d2 = np.abs(a2 * x + b2 * y + c2) / const2  # distance to the second axis
        g2 = np.exp(-d2 ** 2 / (2 * sigma2 ** 2))  # gaussian distribution along the second axis
        g = g1 * g2  # gaussian heatmap

        # 4. Mask the heatmap with box boundary
        heatmap[box_y1:box_y2, box_x1:box_x2, label_id] = np.maximum(heatmap[box_y1:box_y2, box_x1:box_x2, label_id], g)

    return heatmap


def calc_patch_weight(gray_patch):
    # otsu thresholding
    gray_patch = np.array(gray_patch)
    thres, binary_patch = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres = int(thres)

    # calculate mean value of pixels less than threshold and greater than threshold
    under_thresh = gray_patch[gray_patch <= thres]
    over_thresh = gray_patch[gray_patch > thres]
    mean_left = np.mean(under_thresh) if len(under_thresh) > 0 else 0
    mean_right = np.mean(over_thresh) if len(over_thresh) > 0 else 0
    contrast_ratio = (mean_right - mean_left) / 32
    contrast_weight = np.power(contrast_ratio, 0.2)
    contrast_weight = np.clip(contrast_weight, 0.2, 1.2)

    # calculate distribution of pixels less than threshold and greater than threshold
    n_under_thresh = len(under_thresh)
    n_over_thresh = len(over_thresh)
    balance_ratio = (n_under_thresh - n_over_thresh) / (n_under_thresh + n_over_thresh)
    balance_weight = 1.2 * np.power(np.cos(np.pi / 3 * balance_ratio), 0.8)
    balance_weight = np.clip(balance_weight, 0.5, 1.2)

    weight = contrast_weight * balance_weight
    return weight
