"""Inspect a frozen single-file graph on fixed training/development examples.

Raw corner channels, crop reachability, heatmap maxima and centroids are
diagnostics only; they do not replace the canonical benchmark metric.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from screen_corner_single import decode_points, patch_boxes, sample_corner_patches
from training.screen_corner_ablation import read_rows


def run(model_path, data, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = torch.jit.load(str(model_path), map_location='cpu').eval()
    validation = read_rows(data/'validation.jsonl')
    # Fixed source ordering, independent of the predictions from this diagnostic.
    selected = [('linyandan_validation', r) for r in read_rows(data/'linyandan_validation.jsonl')]
    selected += [('linyandan_train', r) for r in read_rows(data/'linyandan_train.jsonl')[:5]]
    selected += [('old_validation', r) for r in read_rows(data/'old_validation.jsonl')[:5]]
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    records = []
    for subset, row in selected:
        bgr = cv2.imread(row['image_path'])
        if bgr is None:
            raise FileNotFoundError(row['image_path'])
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255.
        truth = np.array(row['manual_quad']) / [w,h]
        diag = np.linalg.norm(np.ptp(np.array(row['manual_quad']), axis=0))
        with torch.inference_mode():
            small = F.interpolate(tensor, [256,256], mode='bilinear', align_corners=False)
            logits, boundary, feature = model.global_head(model.encoder((small-model.mean)/model.std))
            coarse = decode_points(logits)
            boxes = patch_boxes(coarse)
            patches = sample_corner_patches(tensor, boxes, 64)
            context = sample_corner_patches(feature, boxes, 16)
            relative, heatmaps = model.local_head(patches, context)
            centroid = decode_points(heatmaps)[:,0]
            final = (boxes[...,:2]+relative.view(1,4,2)*(boxes[...,2:]-boxes[...,:2])).clamp(0.,1.)
            torch.testing.assert_close(final,model(tensor)['final_quad'])
        boxes = boxes[0].numpy(); coarse = coarse[0].numpy(); final = final[0].numpy()
        centroid = centroid.numpy(); heatmaps=heatmaps[:,0].numpy()
        local_truth = (truth-boxes[:,:2])/(boxes[:,2:]-boxes[:,:2])
        canvas = Image.new('RGB',(1280,1120),'#f4f5f7')
        draw = ImageDraw.Draw(canvas)
        draw.text((12,8),f"{Path(row['image_path']).stem} | {subset} | {model_path.parent.name}",font=font,fill='#17212b')
        thumb=Image.fromarray(rgb).resize((620,390))
        d=ImageDraw.Draw(thumb)
        for pts,color in [(truth,'#00ed38'),(coarse,'#29b6ff'),(final,'#ff3030')]:
            xy=[(float(p[0])*620,float(p[1])*390) for p in pts]
            d.line(xy+[xy[0]],fill=color,width=3)
        for box in boxes:d.rectangle((box[0]*620,box[1]*390,box[2]*620,box[3]*390),outline='#ffdf00',width=1)
        canvas.paste(thumb,(10,40))
        # Predicted edge probability over the full source; no human edge input.
        edge=boundary[0,1].sigmoid().numpy()
        edge=cv2.resize(edge,(620,390)); heat=cv2.applyColorMap(np.uint8(edge*255),cv2.COLORMAP_TURBO)[:,:,::-1]
        overlay=Image.fromarray(np.uint8(.6*np.asarray(thumb)+.4*heat))
        canvas.paste(overlay,(650,40))
        draw.text((12,438),'green=GT blue=coarse red=final yellow=crop',font=font,fill='#17212b')
        draw.text((650,438),'Predicted boundary probability (full target)',font=font,fill='#17212b')
        corners=[]
        for i,label in enumerate(['TL','TR','BR','BL']):
            y=480+i*154
            patch=np.uint8(patches[i].numpy().transpose(1,2,0)*255)
            weights=torch.softmax(torch.from_numpy(heatmaps[i]).flatten()*12,0).numpy().reshape(16,16)
            peak_index=np.unravel_index(np.argmax(weights),weights.shape)
            peak=np.array([peak_index[1]/15,peak_index[0]/15])
            peak_point=boxes[i,:2]+peak*(boxes[i,2:]-boxes[i,:2])
            exp_point=boxes[i,:2]+centroid[i]*(boxes[i,2:]-boxes[i,:2])
            errors={name:float(np.linalg.norm((point-truth[i])*[w,h])/diag) for name,point in [('peak',peak_point),('expectation',exp_point),('final',final[i])]}
            # Count separated competing responses as a diagnostic, without claiming causation.
            remaining=weights.copy(); yy,xx=np.ogrid[:16,:16]
            remaining[(xx-peak_index[1])**2+(yy-peak_index[0])**2<=4]=0
            secondary=float(remaining.max()/max(weights.max(),1e-12))
            inside=bool(((local_truth[i]>=0)&(local_truth[i]<=1)).all())
            corners.append(dict(corner=label,truth_in_crop=inside,errors=errors,
                                peak=peak.tolist(),centroid=centroid[i].tolist(),
                                competing_peak_ratio=secondary))
            for col,show_heat in enumerate([False,True]):
                p=Image.fromarray(patch).resize((280,140))
                if show_heat:
                    heat=cv2.applyColorMap(np.uint8(weights/max(weights.max(),1e-12)*255),cv2.COLORMAP_TURBO)[:,:,::-1]
                    heat=Image.fromarray(heat).resize((280,140),Image.Resampling.NEAREST)
                    p=Image.blend(p,heat,.55)
                pd=ImageDraw.Draw(p)
                for q,color in [(local_truth[i],'#00ff38'),(peak,'#ffdf00'),(centroid[i],'#00ffff'),(relative[i].numpy(),'#ff3030')]:
                    x0,y0=q*[280,140];pd.ellipse((x0-3,y0-3,x0+3,y0+3),outline=color,width=2)
                canvas.paste(p,(12+col*296,y))
            draw.text((610,y+4),f'{label} GT in crop={inside}',font=font,fill='#17212b')
            draw.text((610,y+32),f"error peak={errors['peak']*100:.2f}% exp={errors['expectation']*100:.2f}% final={errors['final']*100:.2f}%",font=font,fill='#17212b')
            draw.text((610,y+60),f'competing response / max={secondary:.3f}',font=font,fill='#17212b')
            draw.text((610,y+88),'GT green | peak yellow | expectation cyan | final red',font=font,fill='#17212b')
        name=f"{subset}_{Path(row['image_path']).stem}.jpg"; canvas.save(output/name,quality=90)
        records.append(dict(subset=subset,image_path=row['image_path'],sheet=str(output/name),corners=corners))
    write={'policy':'fixed source-order train/dev only, raw channels; no candidate selection or new threshold',
           'model':str(model_path),'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
           'rows':records}
    (output/'diagnostics.json').write_text(json.dumps(write,ensure_ascii=False,indent=2))
    lines=['# 固定训练/开发页热图诊断','','每页展示人工/粗框/最终框、局部裁剪、预测边界、局部热图、主峰与空间期望。仅作原因诊断，不能用替代预测的统计来重新选择候选。','']
    for r in records:lines.extend([f"![{r['subset']} {Path(r['image_path']).stem}](<{r['sheet']}>)",''])
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'pages':len(records),'output':str(output)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.model.resolve(),a.data.resolve(),a.output.resolve())
