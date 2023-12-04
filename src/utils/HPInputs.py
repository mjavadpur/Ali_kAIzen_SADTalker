
class HPInputs:
    def __init__(self, 
                 driven_audio : str= './examples/driven_audio/bus_chinese.wav',
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
                 old_version : bool = True):
        self.driven_audio = driven_audio  # path to driven audio
        self.source_image = source_image  # path to source image
        self.ref_eyeblink = ref_eyeblink  # path to reference video providing eye blinking
        self.ref_pose = ref_pose          #  path to reference video providing pose
        self.checkpoint_dir = checkpoint_dir # path to pretrained models
        self.result_dir = result_dir      # path to output
        self.pose_style = pose_style      # input pose style from [0, 46)
        self.batch_size = batch_size      # the batch size of facerender
        self.size = size                  # the image size of the facerender
        self.expression_scale = expression_scale # 
        self.input_yaw = input_yaw        # the input yaw degree of the user
        self.input_pitch = input_pitch    # the input pitch degree of the user
        self.input_roll = input_roll      # the input roll degree of the user
        self.enhancer = enhancer          # Face enhancer, [gfpgan, RestoreFormer]
        self.background_enhancer = background_enhancer # background enhancer, [realesrgan]
        self.cpu = cpu                    # 
        self.face3dvis = face3dvis        # generate 3d face and 3d landmarks
        self.still = still                # can crop back to the original videos for the full body aniamtion
        self.preprocess = preprocess      # how to preprocess the images. choices=['crop', 'extcrop', 'resize', 'full', 'extfull']
        self.verbose = verbose            # saving the intermedia output or not
        self.old_version = old_version    # use the pth other than safetensor version