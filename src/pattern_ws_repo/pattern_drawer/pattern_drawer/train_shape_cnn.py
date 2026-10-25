"""Train the drawing-shape CNN on synthetic top-down images.

  ros2 run pattern_drawer train_shape_cnn            # about 5-10 min on a laptop CPU
  ros2 run pattern_drawer train_shape_cnn --per-class 800 --epochs 6   # quicker

Images are generated to look like scan_topdown.png (tan table, black pen lines,
1 px ~ 1 mm, small perspective error, blur, noise), so no manual labelling is needed.
Saves the model to ~/pattern_models/shape_cnn.pt and an example sheet of training images.
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
from torch import nn

from pattern_drawer.shape_cnn import CLASSES, ShapeCNN, ink_mask, render_shape, to_tensor


def make_set(per_class, seed):
    rng = random.Random(seed)
    np.random.seed(seed)
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        for _ in range(per_class):
            t = to_tensor(ink_mask(render_shape(name, rng)))
            if t is not None:
                xs.append(t)
                ys.append(label)
    return torch.stack(xs), torch.tensor(ys)


def save_examples(path, rng):
    tiles = []
    for name in CLASSES:
        img = cv2.resize(render_shape(name, rng), (120, 120))
        cv2.putText(img, name, (3, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(img)
    cv2.imwrite(path, np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--out', default=os.path.expanduser('~/pattern_models/shape_cnn.pt'))
    args, _ = ap.parse_known_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    print(f'Generating {args.per_class} training images per class ({len(CLASSES)} classes) ...')
    x_tr, y_tr = make_set(args.per_class, seed=1)
    x_va, y_va = make_set(max(100, args.per_class // 5), seed=2)
    save_examples(os.path.join(os.path.dirname(args.out), 'training_examples.png'),
                  random.Random(3))

    model = ShapeCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = x_tr[idx]
            if random.random() < 0.5:                       # mirror augmentation
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            loss = loss_fn(model(xb), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_va).argmax(1)
        acc = (pred == y_va).float().mean().item() * 100
        print(f'epoch {epoch:2d}/{args.epochs}  loss {total / len(x_tr):.3f}  '
              f'validation accuracy {acc:.1f}%')

    # per-class accuracy on the validation set
    with torch.no_grad():
        pred = model(x_va).argmax(1)
    for label, name in enumerate(CLASSES):
        m = y_va == label
        print(f'  {name:14s} {(pred[m] == label).float().mean().item() * 100:5.1f}%')

    torch.save({'state_dict': model.state_dict(), 'classes': CLASSES}, args.out)
    print(f'Saved model: {args.out}')


if __name__ == '__main__':
    main()