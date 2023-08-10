import torch
from termcolor import colored

from config_parser import config_parser
from nerf_utils import check_assertions
from train import train
from test import test



if __name__=='__main__':
    
    try:
        torch.multiprocessing.set_start_method('spawn') 
    except RuntimeError:
        pass
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    parser = config_parser()
    args = parser.parse_args()
    
    check_assertions(args)
    
    
    if not args.test_mode:
        train(args)
        
    else:
        if args.render_spiral:
            print(colored('****RENDERING SPIRAL SEQUENCE****', 'yellow'))
            test(args, test_type='spiral')

        if args.render_iden_interpolation:
            print(colored('****RENDERING IDENTITY INTERPOLATION SEQUENCE****', 'yellow'))
            test(args, test_type='iden')
        
        if args.render_pose_interpolation:
            print(colored('****RENDERING POSE INTERPOLATION SEQUENCE****', 'yellow'))
            test(args, test_type='pose')
        
        if args.render_shape_variation:
            print(colored('****RENDERING SHAPE VARIATION SEQUENCE****', 'yellow'))
            test(args, test_type='shape')
        
        if args.render_val:
            print(colored('****RENDERING VALIDATION SET****', 'yellow'))
            test(args, test_type='val')
        
        if args.render_custom:
            print(colored('****RENDERING CUSTOM SEQUENCE****', 'yellow'))
            test(args, test_type='custom')
        
        if args.extract_mesh:
            print('Extracting mesh')
            test(args, test_type='mesh')