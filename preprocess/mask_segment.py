"""
Interactive SAM-based object segmentation.

Click on an object in the displayed image to prompt SAM with a positive
point; each click refines the current mask. Press 'a' to save the current
mask to `<image_name>.txt` (append mode) and start segmenting the next
object in the same image.

Usage:
    python mask_segment.py --image path/to/image.jpg --checkpoint path/to/sam_vit_h_4b8939.pth
"""
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*",
               s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*",
               s=marker_size, edgecolor="white", linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


class ImageSegment:
    """Click-to-segment interaction handler backed by a SAM predictor."""

    def __init__(self, ax, predictor, img_path):
        self.ax = ax
        self.predictor = predictor
        self.img_path = img_path
        self.point_label = [1]
        self.mask_input = None
        self.points = np.empty(shape=(0, 2))
        self.mask = None
        self.cid = ax.figure.canvas.mpl_connect("button_press_event", self)
        self.sid = ax.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def __call__(self, event):
        x, y = event.xdata, event.ydata
        self.points = np.concatenate((self.points, np.array([[x, y]])), axis=0)
        input_label = np.array(self.point_label)

        masks, scores, logits = self.predictor.predict(
            point_coords=self.points,
            point_labels=input_label,
            mask_input=self.mask_input,
            multimask_output=True,
        )

        self.mask_input = logits[np.argmax(scores), :, :][None, :, :]
        mask = masks[np.argmax(scores), :, :] if len(masks) > 1 else masks

        show_mask(mask, self.ax)
        show_points(self.points, input_label, self.ax)
        self.mask = mask

        self.ax.figure.canvas.draw()
        self.point_label.append(1)

    def on_key(self, event):
        if event.key == "a":
            mask_name = self.img_path.split("/")[-1].rsplit(".", 1)[0]
            with open(mask_name + ".txt", "a+") as f:
                np.savetxt(f, self.mask, delimiter=",")
            self.point_label = [1]
            self.mask_input = None
            self.points = np.empty(shape=(0, 2))


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM object segmentation.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to the SAM checkpoint (e.g. sam_vit_h_4b8939.pth).")
    parser.add_argument("--model_type", type=str, default="vit_h",
                         choices=["vit_h", "vit_l", "vit_b"], help="SAM backbone type.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM on.")
    args = parser.parse_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.imshow(image)
    ax.axis("off")

    print("Click on an object to segment it. Press 'a' to save the mask and continue.")
    ImageSegment(ax, predictor, args.image)
    plt.show()


if __name__ == "__main__":
    main()
