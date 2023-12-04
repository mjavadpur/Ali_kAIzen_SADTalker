from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


def execute(driven_audio : str= './examples/driven_audio/bus_chinese.wav',
            source_image : str = './examples/source_image/full_body_1.png',
            ref_eyeblink : str = None,
            ref_pose : str = None,
            checkpoint_dir : str = './checkpoints',
            result_dir : str = './results',
            pose_style : int = 0,
            batch_size : int  = 2,
            size : int = 256,
            expression_scale : float = 1.,
            input_yaw : int = None,
            input_pitch : int = None,
            input_roll : int = None,
            enhancer : str = None,
            background_enhancer : str = None,
            cpu : bool = True,
            face3dvis : bool = True,
            still : bool = True,
            preprocess : str = 'crop',
            verbose : bool = True,
            old_version : bool = True,
            # net_recon : str = 'resnet50',
            # use_last_fc : str = None,
            # bfm_folder : str = './checkpoints/BFM_Fitting/',
            # bfm_model : str = 'BFM_model_front.mat',
            # focal : float = 1015.,
            # center : float = 112.,
            # camera_d : float = 10.,
            # z_near : float = 5.,
            # z_far : float = 15.
            )-> str:
    
    pic_path = source_image
    audio_path = driven_audio
    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    input_yaw_list = input_yaw
    input_pitch_list = input_pitch
    input_roll_list = input_roll
    
    parser = ArgumentParser()  
    
    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()


    device = "cpu"
    if torch.cuda.is_available() and not cpu:
        device = "cuda"

    current_root_path = os.path.split(sys.argv[0])[0]
    
    init_path_startTime = time.time()
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)
    
    init_path_endTime = time.time()    
    execution_time = init_path_endTime - init_path_startTime
    print(f"init_path Execution Time: {execution_time} seconds")

    #init model
    CropAndExtract_startTime = time.time()
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    CropAndExtract_endTime = time.time()    
    execution_time = CropAndExtract_endTime - CropAndExtract_startTime
    print(f"CropAndExtract Execution Time: {execution_time} seconds")


    Audio2Coeff_startTime = time.time()
    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    Audio2Coeff_endTime = time.time()    
    execution_time = Audio2Coeff_endTime - Audio2Coeff_startTime
    print(f"Audio2Coeff Execution Time: {execution_time} seconds")
    
    AnimateFromCoeff_startTime = time.time()
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    AnimateFromCoeff_endTime = time.time()    
    execution_time = AnimateFromCoeff_endTime - AnimateFromCoeff_startTime
    print(f"AnimateFromCoeff Execution Time: {execution_time} seconds")

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    generate_startTime = time.time()
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"CropAndExtract preprocess_model.generate Execution Time: {execution_time} seconds")
    
    
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    generate_startTime = time.time()
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"CropAndExtract ref_eyeblink preprocess_model.generate Execution Time: {execution_time} seconds")

    generate_startTime = time.time()
    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"CropAndExtract ref_pose preprocess_model.generate Execution Time: {execution_time} seconds")
    #audio2ceoff
    generate_startTime = time.time()
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still= still)
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"get_data Execution Time: {execution_time} seconds")
    
    
    generate_startTime = time.time()
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"audio_to_coeff.generate Execution Time: {execution_time} seconds")

    # 3dface render
    generate_startTime = time.time()
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"gen_composed_video Execution Time: {execution_time} seconds")
    
    #coeff2video
    generate_startTime = time.time()
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale= expression_scale, still_mode= still, preprocess= preprocess, size= size)
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"get_facerender_data Execution Time: {execution_time} seconds")
    
    
    generate_startTime = time.time()
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer= enhancer, background_enhancer= background_enhancer, preprocess= preprocess, img_size= size)
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"animate_from_coeff.generate Execution Time: {execution_time} seconds")
    
    
    generate_startTime = time.time()
    shutil.move(result, save_dir+'.mp4')
    generate_endTime = time.time()    
    execution_time = generate_endTime - generate_startTime
    print(f"shutil.move Execution Time: {execution_time} seconds")
    
    print('The generated video is named:', save_dir+'.mp4')

    if not verbose:
        shutil.rmtree(save_dir)
        
    return save_dir
