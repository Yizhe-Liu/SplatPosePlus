import os
import torch
import yaml
import json
import pycolmap
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
from shutil import rmtree
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.spatial.transform import Rotation 
from PIL import Image
from gaussian_splatting.train import *
from gaussian_splatting.scene.colmap_loader import read_points3D_binary
from gaussian_splatting.scene.dataset_readers import storePly
from gaussian_splatting.scene.cameras import Camera

# needed for PAD code
from easydict import EasyDict
from utils_pose_est import ModelHelper, update_config, DefectDataset
from aupro import calculate_au_pro_au_roc

# modified hloc to save 
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation, localize_sfm
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer 

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

pre_parser = ArgumentParser(description="Parameters of the LEGO training run")
pre_parser.add_argument("-c", "-classname", metavar="c", type=str, help="current class to run experiments on", default="01Gorilla")
pre_parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("-iters", type=int, help="number of training iterations for 3DGS", default=15000)
pre_parser.add_argument("-skip_loc", help='skip localization', action='store_true')                 
pre_parser.add_argument("-skip_train", help='skip training 3dgs', action='store_true')                        
pre_parser.add_argument("-data_path", type=str, help="preprocessed dataset path", default="MAD-Sim/")         
pre_parser.add_argument("-n_match", type=int, default=15, help="num of matches for netvlad image retrieval")    
pre_parser.add_argument("-trainset_ratio", type=float, default=1.0, help="percentage of training samples")     
pre_parser.add_argument("-feature_ext", type=str, default='superpoint', help="feature extractor", choices=["superpoint", "aliked"])                  

lego_args = pre_parser.parse_args()
data_path = os.path.join(lego_args.data_path, lego_args.c)
result_dir = os.path.join(f"results", lego_args.c)
print("saving model to: ", result_dir)
os.makedirs(result_dir, exist_ok=True)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

loc_time, sfm_time = 0, 0
if not lego_args.skip_loc: # create sfm & localize testset
    scene_path = Path(data_path)
    image_path = scene_path/'images'
    outputs = scene_path/'outputs'
    if os.path.exists(image_path): rmtree(image_path)
    os.makedirs(image_path)
    os.symlink('../train/', image_path/'train/')
    os.symlink('../test/', image_path/'test/')
    ref_pairs = outputs/'ref_pairs.txt'
    loc_pairs = outputs/'loc_pairs.txt'
    os.makedirs(outputs, exist_ok=True)
    with open(os.path.join(scene_path, 'transforms.json')) as f: flist = json.load(f)
    first_img = Image.open(scene_path/'train/good/0.png')
    (w, h), cam_angle_x = first_img.size, flist['camera_angle_x']
    focal = 0.5*w/np.tan(0.5*cam_angle_x)
    camera_id = 1  # increment for each new camera
    rec = pycolmap.Reconstruction()
    rec.add_camera(pycolmap.Camera(model='SIMPLE_PINHOLE', width=w, height=h, params=[focal, w/2, h/2], camera_id=camera_id))
    for i, frame in enumerate(tqdm(flist['frames'])):
        c2w = np.array(frame["transform_matrix"])   # NeRF 'transform_matrix' is a camera-to-world transform
        c2w[:3, 1:3] *= -1  # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.linalg.inv(c2w)
        pose = pycolmap.Rigid3d(w2c[:3])
        im = pycolmap.Image(f'train/good/{i}.png', [], pose, camera_id, i+1)
        im.registered = True
        rec.add_image(im)
        frame["file_path"] = frame["file_path"][:-4]
    rec.write(outputs)
    with open(os.path.join(scene_path, 'transforms_train.json'), 'w') as f: f.write(json.dumps(flist))
    flist["frames"] = flist["frames"][::20]
    with open(os.path.join(scene_path, 'transforms_test.json'), 'w') as f: f.write(json.dumps(flist))

    query_list_with_intrinsics_path = scene_path / 'query_list_with_intrinsics.txt'
    intrinsics_str = f' SIMPLE_PINHOLE {w} {h} {focal} {w/2} {h/2}\n'
    with open(query_list_with_intrinsics_path, 'w') as fp:
        query_list_with_intrinsics = [x + intrinsics_str for x in glob('test/*/*.png', root_dir=image_path)]
        fp.writelines(query_list_with_intrinsics)

    retrieval_conf = extract_features.confs["netvlad"]
    if lego_args.feature_ext == 'superpoint':
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
    else:
        feature_conf = extract_features.confs["aliked-n16"]
        matcher_conf = match_features.confs["aliked+lightglue"]

    start.record()
    global_descriptors = extract_features.main(retrieval_conf, image_path, outputs)
    features = extract_features.main(feature_conf, image_path, outputs)
    pairs_from_retrieval.main(global_descriptors, ref_pairs, num_matched=lego_args.n_match, query_prefix='train', db_prefix='train')
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=lego_args.n_match, query_prefix='test', db_prefix='train')
    sfm_matches = match_features.main(matcher_conf, ref_pairs, feature_conf["output"], outputs)
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf["output"], outputs)
    rec = triangulation.main(outputs, outputs, image_path, ref_pairs, features, sfm_matches)
    end.record()
    sfm_time = start.elapsed_time(end)
    rec.write(outputs)
    xyz, rgb, _ = read_points3D_binary(outputs/'points3D.bin')
    storePly(Path(data_path)/'points3d.ply', xyz, rgb) # store ply for 3DGS
    results = scene_path/'query_poses.txt'

    start.record()
    localize_sfm.main(
        rec,
        query_list_with_intrinsics_path,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  
    end.record()
    loc_time = start.elapsed_time(end)/len(query_list_with_intrinsics)

train_time = 0 
if lego_args.skip_train: print("skipping training!")
else:
    # Set up command line argument parser
    training_args = ["-w", "-s", data_path, "-m", result_dir, "--iterations", str(lego_args.iters), 
                     "--densification_interval", "1000"] 
    print("training args: ", training_args)
    parser = ArgumentParser(description="3DGS Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[lego_args.iters])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[lego_args.iters])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(training_args)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    start.record()
    safe_state(args.quiet, lego_args.seed)  # Initialize system state (RNG)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, lego_args.trainset_ratio)
    end.record()
    torch.cuda.synchronize()
    train_time = start.elapsed_time(end)

    
def syn_pseudo_ref_imgs(cur_class, model_dir, data_dir):
    trainset = DefectDataset(data_dir, cur_class, "train", True, True)
    train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (400,400)), 1, 3).numpy()
    testset = DefectDataset(data_dir, cur_class, "test", True, True)
    fov, tanfov = trainset.camera_angle, np.tan(trainset.camera_angle/2)
    bg = torch.tensor([1.0,1.0,1.0], device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ref_imgs, syn_imgs, all_labels, gt_masks, times, test_RT = [], [], [], [], 0, {}

    with open(Path(data_dir)/cur_class/'query_poses.txt') as fp: 
        for frame in fp.readlines(): # colmap format
            name, qw, qx, qy, qz, x, y, z = frame.split()
            qw, qx, qy, qz, x, y, z = map(float, [qw, qx, qy, qz, x, y, z])
            test_RT[name] = (Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T, [x,y,z]) # R is stored transposed due to 'glm' in CUDA code

    with torch.no_grad():
        gs = GaussianModel(3)
        gs.load_ply(os.path.join(model_dir, f"point_cloud/iteration_{lego_args.iters}/point_cloud.ply"))
        means3D, shs, scales, rotations, opacities = gs.get_xyz, gs.get_features, gs.get_scaling, gs.get_rotation, gs.get_opacity
        for i in tqdm(range(len(testset)), desc='Rendering test views'):
            start.record()
            im_key = '/'.join(testset.images[i].split('/')[2:])
            img, label, mask = testset[i]
            ref_imgs.append(img.cuda())
            all_labels.append(label)
            gt_masks.append(mask)
            cur_view = Camera(1, *test_RT[im_key], FoVx=fov, FoVy=fov, image=img, gt_alpha_mask=None, image_name="", uid=1)
            raster_settings = GaussianRasterizationSettings(    # Set up rasterization configuration
                image_height=cur_view.image_height,
                image_width=cur_view.image_width,
                tanfovx=tanfov,
                tanfovy=tanfov,
                bg=bg,
                scale_modifier=1,
                viewmatrix=cur_view.world_view_transform,
                projmatrix=cur_view.full_proj_transform,
                sh_degree=3,
                campos=cur_view.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            syn_imgs.append(rasterizer(means3D, None, opacities, shs, None, scales, rotations, None)[0])
            end.record()
            torch.cuda.synchronize()
            times += start.elapsed_time(end)
    return syn_imgs, ref_imgs, all_labels, gt_masks, times/len(testset)

syn_imgs, ref_imgs, all_labels, gt_masks, nvs_time = syn_pseudo_ref_imgs(lego_args.c, result_dir, lego_args.data_path)

# load pre-trained effnet
with open("PAD_utils/config_effnet.yaml") as f:
    mad_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
M = ModelHelper(update_config(mad_config).net)
M.eval()
M.cuda()

# Evaluation code using v2 api & batched ops 
# Adapted from PAD/MAD data set paper at https://github.com/EricLee0224/PAD
CR = torch.nn.MSELoss(reduction='none')
tf_img = v2.Compose([v2.Resize(224), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
upscale = v2.Resize(224)
start.record()
with torch.no_grad():
    ref, syn = tf_img(torch.stack(ref_imgs)), tf_img(torch.stack(syn_imgs)) # n by 3 by 224 by 224
    scores = v2.GaussianBlur(9, 4)(CR(ref, syn).sum(1) + sum([upscale(CR(*x).sum(1)) for x in zip(M(ref), M(syn))]))
end.record()
torch.cuda.synchronize()
cnn_time = start.elapsed_time(end)/len(syn_imgs)
scores_min = scores.min()
scores = ((scores - scores_min) / (scores.max() - scores_min)).cpu().numpy()
gt_mask = v2.Resize(224, interpolation=v2.InterpolationMode.NEAREST)(torch.stack(gt_masks)).cpu().numpy()
precision, recall, thresholds = precision_recall_curve(gt_mask.ravel(), scores.ravel())
a, b = 2 * precision * recall, precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]
fpr, tpr, _ = roc_curve(gt_mask.ravel(), scores.ravel())
per_pixel_rocauc = roc_auc_score(gt_mask.ravel(), scores.ravel())
au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(gt_mask, scores)
img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list_isano = np.asarray(all_labels) != 0
img_roc_auc = roc_auc_score(gt_list_isano, img_scores)
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
print(f"aupro: {au_pro}. and other au_roc: {au_roc}")
print('image ROCAUC: %.3f' % (img_roc_auc))

with open(Path(result_dir)/f'metrics_{lego_args.seed}_{lego_args.trainset_ratio}.txt','w') as fp:
    fp.write(f'Pixel_ROCAUC: {per_pixel_rocauc}\naupro: {au_pro}\nau_roc: {au_roc}\nImage_ROCAUC: {img_roc_auc}\n'
            f'SFM_time: {sfm_time}\nLoc_time: {loc_time}\nTrain_time: {train_time}\nNVS_time: {nvs_time}\nCNN_time: {cnn_time}')