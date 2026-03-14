import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from FusionModelDataset import FusionModelDataset
from PST900Dataset import PST900Dataset, get_pst900_palette
from FusionModelUtils import compute_results,get_palette,visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
import torch.nn.functional as F 
from FusionModel import FusionModel
from FusionModelTrain import ModelEMA
from tent import Tent,collect_params,configure_model

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='FusionModel')
parser.add_argument('--weight_name', '-w', type=str, default='checkpoints')
parser.add_argument('--file_name', '-f', type=str, default='best_model.ema.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--have_test_labels', '-htl', type=bool, default=True) 
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480) 
parser.add_argument('--img_width', '-iw', type=int, default=640)  
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--context_dim', type=str, default='[96,192,384,768]')
parser.add_argument('--dataset', type=str, default='mfnet', choices=['mfnet', 'pst900'],
                    help='Dataset to evaluate: mfnet or pst900')
parser.add_argument('--data_dir', '-dr', type=str, default='/datavolume/data/emrecanitez/Datasets/MFNet')
parser.add_argument('--model_dir', '-wd', type=str, default='Experiments/experiment_sigma_killer_s2') #'/datavolume/data/emrecanitez/experiments/Experiment'
#############################################################################################
# ---- TTA argümanları ----
parser.add_argument('--tta', action='store_true', default=False,
                    help='Enable test-time augmentation')
parser.add_argument('--tta_scales', type=float, nargs='+', default=[1.0],
                    help='TTA scales, e.g. 0.75 1.0 1.25')
parser.add_argument('--tta_flip', action='store_true', default=False,
                    help='Add horizontal flip to TTA')

# ---- TENT argümanları ----
parser.add_argument('--use_tent', action='store_true', default=False, help='Enable TENT test-time adaptation')

# ---- Decoder/Fusion architecture ----
parser.add_argument('--rgb_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                    help='timm model name for RGB backbone')
parser.add_argument('--ir_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                    help='timm model name for IR backbone')
parser.add_argument('--decoder_type', type=str, default='fpn', choices=['fpn', 'panet'],
                    help='Decoder type: fpn or panet')
parser.add_argument('--deep_supervision', action='store_true', default=False,
                    help='Enable deep supervision for PANet decoder')
parser.add_argument('--enhanced_fusion', action='store_true', default=False,
                    help='Use EnhancedSemanticFusion for stages 3-4')
parser.add_argument('--use_safd', action='store_true', default=False,
                    help='Novel: Scene-Adaptive Frequency Decomposition')
parser.add_argument('--use_cafg', action='store_true', default=False,
                    help='Novel: Complementarity-Aware Fusion Gate')
parser.add_argument('--use_tpsw', action='store_true', default=False,
                    help='Novel: Thermal Prior-Guided Spatial Weighting')

args = parser.parse_args()
#############################################################################################

def tta_inference(model, rgb, ir, scales=[1.0], do_flip=False, output_size=(480, 640)):
    """
    Apply test-time augmentation (TTA) by scaling and flipping.
    Args:
        model: segmentation model
        rgb: RGB input tensor [1, C, H, W]
        ir: IR input tensor [1, C, H, W]
        scales: list of scale factors (e.g., [0.75, 1.0, 1.25])
        do_flip: whether to apply horizontal flip
        output_size: desired final output size (height, width)
    Returns:
        Averaged logits after TTA
    """
    prob_list = []

    for scale in scales:
        # --- Resize inputs ---
        if scale != 1.0:
            # We must ensure new dimensions are divisible by 32 for ConvNeXt
            new_h = int(rgb.shape[2] * scale / 32) * 32
            new_w = int(rgb.shape[3] * scale / 32) * 32
            rgb_scaled = F.interpolate(rgb, size=(new_h, new_w), mode='bilinear', align_corners=False)
            ir_scaled  = F.interpolate(ir,  size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            rgb_scaled = rgb
            ir_scaled = ir

        # --- Normal forward ---
        logits, _ = model(rgb_scaled, ir_scaled)
        logits = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
        prob = F.softmax(logits, dim=1)
        prob_list.append(prob)

        # --- Horizontal flip ---
        if do_flip:
            rgb_flipped = torch.flip(rgb_scaled, dims=[3])
            ir_flipped  = torch.flip(ir_scaled, dims=[3])
            logits_flip, _ = model(rgb_flipped, ir_flipped)
            logits_flip = torch.flip(logits_flip, dims=[3])  # flip back
            logits_flip = F.interpolate(logits_flip, size=output_size, mode='bilinear', align_corners=False)
            prob_flip = F.softmax(logits_flip, dim=1)
            prob_list.append(prob_flip)

    # --- Average all probabilities ---
    final_prob = torch.stack(prob_list, dim=0).mean(dim=0)
    return final_prob


if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # prepare save direcotry
    if os.path.exists("./runs"):
        print("previous \"./runs\" folder exist, will delete this folder")
        shutil.rmtree("./runs")
    os.makedirs("./runs")
    os.chmod("./runs", stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine
    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    
    conf_total = np.zeros((args.n_class, args.n_class))
    ctx_dim = [int(x.strip()) for x in args.context_dim.strip('[]').split(',')]
    resolution = (args.img_height, args.img_width)
    model = eval(args.model_name)(
        rgb_arch=args.rgb_arch,
        ir_arch=args.ir_arch,
        num_classes=args.n_class,
        context_dim=ctx_dim,
        input_resolution=resolution,
        rgb_backbone_resolution=resolution,
        ir_backbone_resolution=resolution,
        output_resolution=resolution,
        decoder_type=args.decoder_type,
        deep_supervision=args.deep_supervision,
        enhanced_fusion=args.enhanced_fusion,
        use_safd=args.use_safd,
        use_cafg=args.use_cafg,
        use_tpsw=args.use_tpsw
    ).to(device)
    print('loading model file %s... ' % model_file)
    checkpoint  = torch.load(model_file,  map_location= device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing: {len(missing)} | unexpected: {len(unexpected)}")

    # if EMA weights exist, load and apply them
    if checkpoint.get('ema_state_dict') is not None:
        ema = ModelEMA(model)
        ema.load_state_dict(checkpoint['ema_state_dict'], device)
        ema.apply_shadow(model) 
        print("[INFO] EMA weights loaded and applied.")
    else:
        print("[INFO] No EMA weights found, using normal weights.")

            
    print(f"Loaded model best mIoU: {checkpoint['best_miou']:.4f}")

    # Load dataset based on --dataset argument
    if args.dataset == 'pst900':
        print(f"[INFO] Loading PST900 dataset ({args.n_class} classes)")
        test_dataset = PST900Dataset(
            data_dir=args.data_dir,
            split=args.dataset_split,  # 'test' for PST900
            rgb_size=resolution,
            thermal_size=resolution,
            have_label=args.have_test_labels,
            use_augmentation=False
        )
    elif args.dataset == 'mfnet':
        print(f"[INFO] Loading MFNet dataset ({args.n_class} classes)")
        test_dataset = FusionModelDataset(
            data_dir=args.data_dir,
            split=args.dataset_split,
            have_label=args.have_test_labels
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    batch_size = 1
    # Create loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    if args.tta:
        mult = 2 if args.tta_flip else 1
        print(f"[INFO] TTA enabled | scales={args.tta_scales} | flip={args.tta_flip} "
              f"| total-forwards-per-image={len(args.tta_scales)*mult}")    
    ave_time_cost = 0.0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    if args.use_tent:
        model = configure_model(model)
        tent_params,param_names = collect_params(model)
        for p, name in zip(tent_params, param_names):
            if not p.requires_grad:
                print(f"[WARN] Param {name} does not have requires_grad=True")

        tent_optimizer = torch.optim.SGD(tent_params, lr=1e-3) 
        tent = Tent(model, tent_optimizer)
        print("[INFO] TENT enabled | optimizing params:", param_names)

    if not args.use_tent:
        model.eval()
    for it, (rgb,ir,labels,img_name) in enumerate(test_loader):
        times = 0.0
        rgb = Variable(rgb).cuda(args.gpu)
        ir = Variable(ir).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        starter.record()
        if args.tta:
            with torch.no_grad():
                logits = tta_inference(model, rgb, ir, scales=args.tta_scales, do_flip=args.tta_flip, output_size=(args.img_height, args.img_width))
                aux = None
        elif args.use_tent:
            logits = tent(rgb, ir)
            aux = None
        else:
            with torch.no_grad():
                logits, aux = model(rgb, ir)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        if it>=5: # # ignore the first 5 frames
            ave_time_cost += curr_time
        # convert tensor to numpy 1d array
        label = labels.cpu().numpy().squeeze().flatten()
        prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
        # generate confusion matrix frame-by-frame (support both MFNet: 9 classes and PST900: 5 classes)
        conf = confusion_matrix(y_true=label, y_pred=prediction, labels=list(range(args.n_class))) # conf is an n_class*n_class matrix
        conf_total += conf
        # save demo images
        #visualize(image_name=[str(img_name)], predictions=logits.argmax(1),weight_name=args.weight_name,rgb=rgb,ir=ir,labels=labels)
        print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                %(args.model_name, args.weight_name, it+1, len(test_loader), str(it), curr_time))

    precision_per_class, recall_per_class, iou_per_class, f1score = compute_results(conf_total)
    conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    savemat(conf_total_matfile,  {'conf': conf_total}) # 'conf' is the variable name when loaded in Matlab
 
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print('* the weight name: %s' %args.weight_name) 
    print('* the file name: %s' %args.file_name)
    print('\n* the per-class recall:')
    for i in range(args.n_class):
        print('    Class %d: %.6f' % (i, recall_per_class[i]))
    print('\n* the per-class iou:')
    for i in range(args.n_class):
        print('    Class %d: %.6f' % (i, iou_per_class[i]))
    print('\n* the per-class f1score:')
    for i in range(args.n_class):
        print('    Class %d: %.6f' % (i, f1score[i]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f, precision: %.6f, f1score: %.6f " \
          %(recall_per_class.mean(), iou_per_class.mean(), precision_per_class.mean(), f1score.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f, precision: %.6f, f1score: %.6f " \
          %(np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class)),precision_per_class.mean(), f1score.mean()))
    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost/(len(test_loader)-5), 1000/(ave_time_cost/(len(test_loader)-5)))) # ignore the first 10 frames
    print('\n###########################################################################')