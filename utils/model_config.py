import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification
from transformers import DistilBertForSequenceClassification
from utils.schedulers import GradualWarmupScheduler
from torchlars import LARS
from utils.general_utils import create_optimizer, create_scheduler

"""
Model configuration and initialization utilities.
"""

def get_models(args, nets, model, models=None):
    """
    Initialize models according to the provided args.
    
    For text datasets (e.g., AGNEWS, IMDB, SST5) where args.model is 'DistilBert' or 'Roberta',
    this function loads the corresponding Hugging Face model. If the method requires OOD detection
    (LFOSA, EOAL, PAL), an additional model (with one extra label) is created along with an extra 
    classification head (model_bc) for EOAL.
    
    For image datasets, the appropriate network architectures are loaded based on args.method.
    
    Args:
        args: arguments object with model parameters
        nets: network module containing model classes
        model: model architecture name
        models: optional dictionary of existing models to extend
        
    Returns:
        Dictionary of models
    """
    # Helper function to move a model to the proper device and optionally wrap with data parallelism.
    def prepare_model(m):
        m = m.to(args.device)
        if args.device != "cpu" and args.data_parallel:
            m = nets.nets_utils.MyDataParallel(m, device_ids=args.gpu)
        return m

    # --------------------- Text Dataset Branch ---------------------
    if args.model in ['DistilBert', 'Roberta']:
        def load_text_model(num_labels):
            # Load the corresponding text model with the specified number of labels.
            if args.model == 'DistilBert':
                return DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', num_labels=int(num_labels), output_hidden_states=True)
            else:  # args.model == 'Roberta'
                return RobertaForSequenceClassification.from_pretrained(
                    'roberta-base', num_labels=int(num_labels), output_hidden_states=True)

        # If the method requires OOD detection (and model_bc for EOAL)
        if args.method in ['LFOSA', 'EOAL', 'PAL']:
            backbone = prepare_model(load_text_model(args.num_IN_class))
            ood_detection = prepare_model(load_text_model(args.num_IN_class + 1))
            models = {'backbone': backbone, 'ood_detection': ood_detection}
            
            if args.method == 'EOAL':
                # Initialize an extra classification module for EOAL.
                model_bc = nets.eoalnet.ResClassifier_MME(
                    num_classes=2 * int(args.num_IN_class), norm=False, input_size=512)
                models['model_bc'] = prepare_model(model_bc)
        else:
            backbone = prepare_model(load_text_model(args.num_IN_class))
            models = {'backbone': backbone}
        return models

    # --------------------- Image Dataset Branch ---------------------
    # For methods that require only a backbone model.
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'VAAL', 'WAAL', 'EPIG', 
                       'EntropyCB', 'CoresetCB', 'AlphaMixSampling', 'noise_stability', 'SAAL', 
                       'VESSAL', 'corelog', 'coremse']:
        backbone = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class, args.im_size))
        models = {'backbone': backbone}

    # TIDAL method: backbone plus an auxiliary module.
    elif args.method == 'TIDAL':
        backbone = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class, args.im_size))
        module = prepare_model(nets.tdnet.TDNet())
        models = {'backbone': backbone, 'module': module}

    # SIMILAR method: backbone with an extra output class.
    elif args.method == 'SIMILAR':
        backbone = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class + 1, args.im_size))
        models = {'backbone': backbone}

    # LL method: backbone using a modified version and an extra loss module.
    elif args.method == 'LL':
        backbone = prepare_model(nets.__dict__[model + '_LL'](args.channel, args.num_IN_class, args.im_size))
        loss_module = prepare_model(nets.__dict__['LossNet'](args.im_size))
        models = {'backbone': backbone, 'module': loss_module}

    # CCAL method: backbone plus two extra networks (semantic and distinctive).
    elif args.method == 'CCAL':
        backbone = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class, args.im_size))
        model_sem = prepare_model(nets.__dict__[model + '_CSI'](args.channel, args.num_IN_class, args.im_size))
        model_dis = prepare_model(nets.__dict__[model + '_CSI'](args.channel, args.num_IN_class, args.im_size))
        models = {'backbone': backbone, 'semantic': model_sem, 'distinctive': model_dis} \
                 if models is None else {**models, 'backbone': backbone}

    # MQNet method: backbone with LL variant, an extra loss module, and a CSI module.
    elif args.method == 'MQNet':
        backbone = prepare_model(nets.__dict__[model + '_LL'](args.channel, args.num_IN_class, args.im_size))
        loss_module = prepare_model(nets.__dict__['LossNet'](args.im_size))
        model_csi = prepare_model(nets.__dict__[model + '_CSI'](args.channel, args.num_IN_class, args.im_size))
        models = {'backbone': backbone, 'module': loss_module, 'csi': model_csi} \
                 if models is None else {**models, 'backbone': backbone, 'module': loss_module}

    # Methods that require OOD detection and possibly an extra classifier for EOAL.
    elif args.method in ['LFOSA', 'EOAL', 'PAL']:
        backbone = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class, args.im_size))
        ood_detection = prepare_model(nets.__dict__[model](args.channel, args.num_IN_class + 1, args.im_size))
        models = {'backbone': backbone, 'ood_detection': ood_detection}
        if args.method == 'EOAL':
            model_bc = prepare_model(nets.eoalnet.ResClassifier_MME(
                num_classes=2 * int(args.num_IN_class), norm=False, input_size=512))
            models['model_bc'] = model_bc

    return models

def get_optim_configurations(args, models):
    """
    Build loss, optimizers, and schedulers in a structured way.
    Differentiates between normal vs. OOD usage, extra modules, etc.
    
    Args:
        args: arguments object with optimizer parameters
        models: dictionary of models
        
    Returns:
        Tuple of (criterion, optimizers, schedulers)
    """
    print("lr: {}, momentum: {}, decay: {}".format(args.lr, args.momentum, args.weight_decay))
    
    # Main criterion
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # Always create optimizer & scheduler for the backbone
    backbone_opt = create_optimizer(args, models['backbone'].parameters())
    backbone_sch = create_scheduler(args, backbone_opt, total_epochs=args.epochs)

    optimizers = {'backbone': backbone_opt}
    schedulers = {'backbone': backbone_sch}

    # -----------------------------------------------------------
    # If we have OOD detection in the model dictionary, we create 
    # an optimizer & scheduler for it. This applies to both text 
    # or image, as long as the method is LFOSA/EOAL/PAL, etc.
    # -----------------------------------------------------------
    if 'ood_detection' in models:
        ood_opt = create_optimizer(args, models['ood_detection'].parameters())
        ood_sch = create_scheduler(args, ood_opt, total_epochs=args.epochs)
        optimizers['ood_detection'] = ood_opt
        schedulers['ood_detection'] = ood_sch

    # EOAL - model_bc with a different lr, e.g. args.lr_model
    if args.method == 'EOAL' and 'model_bc' in models:
        model_bc_opt = create_optimizer(
            args,
            models['model_bc'].parameters(),
            lr=args.lr_model  # use a separate LR if needed
        )
        # EOAL code often uses step-based or custom schedule; 
        # set as needed or re-use the same pattern
        optimizers['model_bc'] = model_bc_opt
        # If you also want a scheduler for model_bc, define it here:
        # sched_bc = create_scheduler(args, model_bc_opt, total_epochs=args.epochs)
        # schedulers['model_bc'] = sched_bc

    # -----------------------------------------------------------
    # TIDAL or LL includes an additional 'module' 
    # -----------------------------------------------------------
    if args.method in ['LL', 'TIDAL'] and 'module' in models:
        module_opt = create_optimizer(args, models['module'].parameters())
        # If you'd like to use a different scheduler type or multi-step:
        # for consistency with original code, we can do MultiStepLR
        module_sch = torch.optim.lr_scheduler.MultiStepLR(
            module_opt, 
            milestones=args.milestone
        )
        optimizers['module'] = module_opt
        schedulers['module'] = module_sch

    # -----------------------------------------------------------
    # CCAL has two extra models: 'semantic' and 'distinctive', 
    # each with warm-up schedulers
    # -----------------------------------------------------------
    if args.method == 'CCAL' and 'semantic' in models and 'distinctive' in models:
        # semantic
        sem_opt = torch.optim.SGD(
            models['semantic'].parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        sem_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            sem_opt, 
            T_max=args.epochs_ccal, 
            eta_min=args.min_lr
        )
        warmup_sem = GradualWarmupScheduler(
            sem_opt, 
            multiplier=10.0, 
            total_epoch=args.warmup, 
            after_scheduler=sem_sched
        )

        # distinctive
        dis_opt = torch.optim.SGD(
            models['distinctive'].parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        dis_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            dis_opt, 
            T_max=args.epochs_ccal, 
            eta_min=args.min_lr
        )
        warmup_dis = GradualWarmupScheduler(
            dis_opt, 
            multiplier=10.0, 
            total_epoch=args.warmup, 
            after_scheduler=dis_sched
        )

        optimizers.update({
            'semantic': sem_opt,
            'distinctive': dis_opt
        })
        schedulers.update({
            'semantic': warmup_sem,
            'distinctive': warmup_dis
        })

    # -----------------------------------------------------------
    # MQNet has 'module' + 'csi' with special LARS usage
    # -----------------------------------------------------------
    if args.method == 'MQNet' and 'module' in models and 'csi' in models:
        # module
        module_opt = create_optimizer(args, models['module'].parameters())
        module_sch = torch.optim.lr_scheduler.MultiStepLR(
            module_opt, 
            milestones=args.milestone
        )

        # csi with LARS
        # We'll create a base SGD first, then wrap it with LARS inside create_optimizer
        csi_sgd = torch.optim.SGD(
            models['csi'].parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=1e-6
        )
        # We can wrap it with LARS. You could do so via create_optimizer if you want:
        # but let's show it manually for clarity:
        csi_opt = LARS(csi_sgd, eps=1e-8, trust_coef=0.001)

        csi_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            csi_opt, 
            T_max=args.epochs_csi
        )
        warmup_csi = GradualWarmupScheduler(
            csi_opt, 
            multiplier=10.0, 
            total_epoch=args.warmup, 
            after_scheduler=csi_sched
        )

        optimizers.update({
            'module': module_opt,
            'csi': csi_opt
        })
        schedulers.update({
            'module': module_sch,
            'csi': warmup_csi
        })

    return criterion, optimizers, schedulers
