
# This implements the GFCS method of the paper "Attacking deep networks with surrogate-based adversarial black-box
#   methods is easy" (https://arxiv.org/abs/2203.08725).
# The code is a heavily adapted version of the implementation of SimBA-ODS (https://github.com/ermongroup/ODS) from the
#   paper "Diversity can be Transferred: Output Diversification for White- and Black-box Attacks"
#   (https://arxiv.org/abs/2003.06878).
# As explained in the GFCS paper, the SimBA-ODS block is used here as a secondary "backup" method, and represents a
#   particularly simple choice. The GFCS context allows for it to be replaced with any comparable method; as the paper
#   suggests, any sensible coimage sampler can be a valid choice. (One might want to consider different weighting
#   schemes in the sampling, including biasing towards the loss gradient.)
# The --GFCS option runs GFCS as described in the paper. The --ODS option runs SimBA-ODS. Using neither option defaults
#   back to SimBA (https://arxiv.org/abs/2003.06878) using the pixel basis.

import argparse

import torch
import torchvision.models as models
import torchvision.datasets as datasets
import numpy as np

import eval_sets
import gfcs_util


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Device for evaluating networks.')
parser.add_argument('--model_name', type=str, required=True, help='Target model to use.')
parser.add_argument('--smodel_name', type=str, nargs='+',
                    help='One or more surrogate models to use (enter all names, separated by spaces).')
parser.add_argument('--targeted', action='store_true', help='If true, perform targeted attack; else, untargeted.')
parser.add_argument('--ODS', action='store_true', help='Perform ODS (original SimBA-ODS).')
parser.add_argument('--GFCS', action='store_true', help='Activate GFCS method.')
parser.add_argument('--num_step', type=int, default=10000, help="Number of 'outer' SimBA iterations. Note that each "
                                                                "iteration may consume 1 or 2 queries.")
parser.add_argument('--num_sample', default=10, type=int, help='Number of sample images to attack.')
parser.add_argument('--data_index_set', type=str,
                    choices=['vgg16_bn_mstr', 'vgg16_bn_batch0', 'vgg16_bn_batch1', 'vgg16_bn_batch2',
                             'vgg16_bn_batch3', 'vgg16_bn_batch4', 'vgg16_bn_batch0_2', 'vgg16_bn_batch3_4',
                             'resnet50_mstr', 'resnet50_batch0', 'resnet50_batch1', 'resnet50_batch2',
                             'resnet50_batch3', 'resnet50_batch4', 'resnet50_batch0_2', 'resnet50_batch3_4',
                             'inceptionv3_mstr', 'inceptionv3_batch0', 'inceptionv3_batch1','inceptionv3_batch2',
                             'inceptionv3_batch3', 'inceptionv3_batch4', 'inceptionv3_batch0_2', 'inceptionv3_batch3_4',
                             'imagenet_val_random'],
                    default='imagenet_val_random',
                    help='The indices from the ImageNet val set to use as inputs. Most options represent predefined '
                         'randomly sampled batches. imagenet_val_random samples from the val set randomly, and may not '
                         'necessarily give images that are correctly classified by the target net.')
parser.add_argument('--step_size', default=0.2, type=float, help='Optimiser step size (as in SimBA).')
parser.add_argument('--output', required=True, help='Name of the output file.')
parser.add_argument('--norm_bound', type=float, default=float('inf'),
                    help='Radius of l2 norm ball onto which solution will be maintained through PGD-type optimisation. '
                         'If not supplied, is effectively infinite (norm is unconstrained).')
parser.add_argument('--net_specific_resampling', action='store_true',
                    help='If specified, resizes input images to match expectations of target net (as always), but adds '
                         'a linear interpolation step to each surrogate network to match its expected resolution. '
                         'Gradients are thus effectively computed in the native surrogate resolutions and returned to '
                         'the target net''s own resolution via the reverse interpolation.')

args = parser.parse_args()

if args.GFCS:
    args.ODS = True  # The code always expects ODS to be activated if GFCS is chosen, so ensure it.

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
mean = gfcs_util.imagenet_mean
std = gfcs_util.imagenet_std

pretrained_model = getattr(models, args.model_name)(pretrained=True)
model = torch.nn.Sequential(
    gfcs_util.Normalise(mean, std),
    pretrained_model
)
model.to(device).eval()

surrogate_model_list = []
for s in range(len(args.smodel_name)):
    pretrained_model = getattr(models, args.smodel_name[s])(pretrained=True)
    if args.net_specific_resampling:
        # Note that this is, by necessity, case-by-case. If using any nets other than inception_v3 that use input
        # resolutions other than 224x224, they must be added here.
        image_width = 299 if args.smodel_name[s] == 'inception_v3' else 224
        pretrained_model = torch.nn.Sequential(
            gfcs_util.Interpolate(torch.Size([image_width, image_width]), 'bilinear'),
            gfcs_util.Normalise(mean, std),
            pretrained_model
        )
    else:
        pretrained_model = torch.nn.Sequential(
            gfcs_util.Normalise(mean, std),
            pretrained_model
        )
    surrogate_model_list.append(pretrained_model.to(device).eval())

loss_func = torch.nn.functional.cross_entropy if args.targeted else gfcs_util.margin_loss

data_transform, image_width = gfcs_util.generate_data_transform(
    "imagenet_inception_299" if args.model_name == "inception_v3" else "imagenet_common_224"
)

# Set your ImageNet folder path here. Consult the documentation for torchvision.datasets.ImageNet to understand what
# files must be placed where initially. Only the val set is required here.
#imagenet_path = '/your/imagenet/dataset/path'
imagenet_path = '../datasets/imagenet_data'
dataset = datasets.ImageNet(imagenet_path, split='val', transform=data_transform)

if args.data_index_set == 'imagenet_val_random':
    input_index_list = torch.randperm(len(dataset))[:args.num_sample]
else:
    input_index_list = getattr(eval_sets, args.data_index_set)[:args.num_sample]

success_list = []
l2_list = []
linf_list = []
queries_list = []

if args.targeted:
    target_class_list = []

if args.GFCS:
    grad_fail_queries = []
    grad_succ_queries = []
    ods_fail_queries = []
    ods_succ_queries = []

if args.ODS and not args.GFCS:
    using_ods = True

for i, s in enumerate(input_index_list):
    (image, label) = dataset[s]
    image.unsqueeze_(0)
    label = torch.LongTensor([label])

    image = image.to(device)
    label = label.to(device)
    label_attacked = label.clone()

    if args.targeted:
        label_attacked[0] = gfcs_util.any_imagenet_id_but(label.item())

    logits = model(image).data
    to_attack = (torch.argmax(logits, dim=1) != label_attacked) if args.targeted else (
            torch.argmax(logits, dim=1) == label_attacked)
    if to_attack:
        X_best = image.clone()
        if args.targeted:
            loss_best = -loss_func(logits, label_attacked)
            class_org = label[0].item()
            class_tgt = label_attacked[0].item()
        else:
            loss_best, class_org, class_tgt = loss_func(logits.data, label_attacked)
        nQuery = 1  # query for the original image

        if args.GFCS:
            n_grad_fail_queries = 0
            n_grad_succ_queries = 0
            n_ods_fail_queries = 0
            n_ods_succ_queries = 0
            using_ods = False
            surrogate_ind_list = torch.randperm(len(surrogate_model_list))

        for m in range(args.num_step):
            if args.ODS:
                X_grad = X_best.detach().clone().requires_grad_()
                if args.GFCS:
                    random_direction = torch.zeros(1, 1000).to(device)
                    random_direction[0, class_org] = -1
                    random_direction[0, class_tgt] = 1
                    if surrogate_ind_list.numel() > 0:
                        ind = surrogate_ind_list[0]
                        surrogate_ind_list = surrogate_ind_list[1:]
                    else:  # You're stuck, so time to revert.
                        random_direction = torch.rand((1, 1000)).to(device) * 2 - 1
                        ind = np.random.randint(len(surrogate_model_list))
                        using_ods = True
                else:
                    random_direction = torch.rand((1, 1000)).to(device) * 2 - 1
                    ind = np.random.randint(len(surrogate_model_list))

                with torch.enable_grad():
                    if args.targeted and not using_ods:
                        # Then you want the target-label x-ent loss from the surrogate:
                        loss = -loss_func(surrogate_model_list[ind](X_grad), label_attacked)
                    else:  # Either margin loss gradient or ODS direction, depending on above context.
                        loss = (surrogate_model_list[ind](X_grad) * random_direction).sum()
                loss.backward()
                delta = X_grad.grad / X_grad.grad.norm()
            else:  # If you're using neither GFCS nor ODS, it falls back to pixel SimBA.
                ind1 = np.random.randint(3)
                ind2 = np.random.randint(image_width)
                ind3 = np.random.randint(image_width)
                delta = torch.zeros(X_best.shape).cuda()
                delta[0, ind1, ind2, ind3] = 1
            for sign in [1, -1]:
                X_pert = X_best - image + (args.step_size * sign * delta)
                if X_pert.norm() > args.norm_bound:
                    X_pert = X_pert / X_pert.norm() * args.norm_bound
                X_new = image + X_pert

                X_new = torch.clamp(X_new, 0, 1)
                logits = model(X_new).data
                nQuery += 1
                if args.targeted:
                    loss_new = -loss_func(logits.data, label_attacked)
                    class_tgt_new = class_tgt  # The target is actually fixed: this is a dummy variable.
                    class_org_new = torch.argmax(logits, dim=1)  # The top finisher can actually change, in a targeted
                    #   attack, but using the x-ent loss on the target class alone, this won't actually matter.
                else:
                    loss_new, class_org_new, class_tgt_new = loss_func(logits.data, label_attacked)
                if loss_best < loss_new:
                    X_best = X_new
                    loss_best = loss_new
                    class_org = class_org_new
                    class_tgt = class_tgt_new
                    if args.GFCS:
                        if using_ods:
                            n_ods_succ_queries += 1
                        else:
                            n_grad_succ_queries += 1
                        # On optimisation success, reset the surrogate list and ensure that you go back to gradients.
                        surrogate_ind_list = torch.randperm(len(surrogate_model_list))
                        using_ods = False
                    break
                # If you reach here, this attempt didn't work, so we count fail queries:
                if args.GFCS:
                    if using_ods:
                        n_ods_fail_queries += 1
                    else:
                        n_grad_fail_queries += 1

            success = (torch.argmax(logits, dim=1) == label_attacked) if args.targeted else (
                    torch.argmax(logits, dim=1) != label_attacked)

            if success:
                print('image %d: attack is successful. query = %d, dist = %.4f' % (
                        i + 1, nQuery, (X_best - image).norm()))
                if args.GFCS:
                    print(f"grad success queries: {n_grad_succ_queries}, grad fail queries: {n_grad_fail_queries}, "
                          f"ODS success queries: {n_ods_succ_queries}, ODS fail queries: {n_ods_fail_queries}")
                break

            if m == args.num_step - 1:
                print('image %d: attack is not successful (query = %d)' % (i + 1, nQuery))
                if args.GFCS:
                    print(f"grad success queries: {n_grad_succ_queries}, grad fail queries: {n_grad_fail_queries}, "
                          f"ODS success queries: {n_ods_succ_queries}, ODS fail queries: {n_ods_fail_queries}")

        success_list.append(success.item())
        queries_list.append(nQuery)
        l2_list.append((X_best - image).norm(p=2).item())
        linf_list.append((X_best - image).norm(p=np.inf).item())
        if args.GFCS:
            grad_fail_queries.append(n_grad_fail_queries)
            grad_succ_queries.append(n_grad_succ_queries)
            ods_fail_queries.append(n_ods_fail_queries)
            ods_succ_queries.append(n_ods_succ_queries)
        if args.targeted:
            target_class_list.append(label_attacked[0].item())

    else:
        print('image %d: already adversary' % (i + 1))

print("Saving to file", args.output)

output_dict = {
    "succs": torch.BoolTensor(success_list),
    "queries": torch.IntTensor(queries_list),
    "l2_norms": torch.as_tensor(l2_list),
    "linf_norms": torch.as_tensor(linf_list),
    "input_args": args
}
if args.GFCS:
    output_dict["grad_succ_queries"] = torch.IntTensor(grad_succ_queries)
    output_dict["grad_fail_queries"] = torch.IntTensor(grad_fail_queries)
    output_dict["ods_succ_queries"] = torch.IntTensor(ods_succ_queries)
    output_dict["ods_fail_queries"] = torch.IntTensor(ods_fail_queries)
if args.targeted:
    output_dict["target_class_list"] = torch.IntTensor(target_class_list)

torch.save(output_dict, args.output)
