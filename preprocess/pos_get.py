"""
Estimate each segmented object's relative position and scale within the
original image, for use as `center_point` / `scale` hints in a scene config.

Reads the `<image_name>.txt` mask file produced by `mask_segment.py` (one
mask per segmented object, stacked along axis 0) and prints, for every
object, a normalized bounding box and its position/scale relative to the
largest ("main") object in the image.

Usage:
    python pos_get.py --image path/to/image.jpg
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_bbox(mask, w, h):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    x1, y1, x2, y2 = w - cmax, h - rmax, w - cmin, h - rmin
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(
        description="Estimate relative object positions/scales from SAM masks."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--show", action="store_true", help="Display the masks over the image.")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    mask_name = args.image.split("/")[-1].rsplit(".", 1)[0]
    masks = np.loadtxt(mask_name + ".txt", delimiter=",").reshape((-1, h, w))

    if args.show:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.imshow(image)
        ax.axis("off")
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        for mask in masks:
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

    # The largest mask is treated as the main/reference object.
    main_object = int(np.argmax([np.count_nonzero(mask) for mask in masks]))
    main_object_pos = np.array([1.0, 1.0])
    main_object_scale = 1.0

    xmin, ymin, xmax, ymax = compute_bbox(masks[main_object], w, h)
    max_len = max(ymax - ymin, xmax - xmin)
    main_object_origin_pos = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

    center_poss, scales = [], []
    for i, mask in enumerate(masks):
        if i == main_object:
            center_poss.append(main_object_pos)
            scales.append(main_object_scale)
            continue

        oxmin, oymin, oxmax, oymax = compute_bbox(mask, w, h)
        object_max_len = max(oymax - oymin, oxmax - oxmin)
        object_scale = object_max_len / max_len * main_object_scale
        object_origin_pos = np.array([(oxmin + oxmax) / 2, (oymin + oymax) / 2])
        object_dis = (object_origin_pos - main_object_origin_pos) / max_len * main_object_scale
        center_poss.append(main_object_pos + object_dis)
        scales.append(object_scale)

    print(f"main object index: {main_object}")
    print("center_poss:", center_poss)
    print("scales:", scales)
    print("\nper-object simulation-domain bounding box "
          "[xmin, xmax, ymin(fixed=1), ymax(fixed=1), zmin, zmax]:")
    for i, c in enumerate(center_poss):
        s = scales[i]
        print(c[0] - s / 2, c[0] + s / 2, 1.0 - s / 2, 1.0 + s / 2, c[1] - s / 2, c[1] + s / 2)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
