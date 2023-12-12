import os, sys
import gradio as gr

# Assuming the SadTalker class is correctly defined in src.gradio_demo
from src.gradio_demo import SadTalker  

def sadtalker_demo(checkpoint_path='/content/kaizen_sadtalker/checkpoints', config_path='/content/kaizen_sadtalker/src/config', warpfn=None):
# def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'>Github</a> </div>")
        
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image")
                driven_audio = gr.Audio(label="Input audio", type="filepath")
                preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'], value='crop', label='Preprocess', info="How to handle input image?")
                is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess 'full')")
                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                batch_size = gr.Slider(label="Batch size in generation", step=1, maximum=10, value=2)
                size_of_image = gr.Radio([256, 512], value=256, label='Face model resolution', info="Use 256/512 model?")
                pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0)
                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

        gen_video = gr.Video(label="Generated video", format="mp4")

        if warpfn:
            submit.click(
                fn=warpfn(sad_talker.test), 
                inputs=[
                    source_image,
                    driven_audio,
                    preprocess_type,
                    is_still_mode,
                    enhancer,
                    batch_size,                            
                    size_of_image,
                    pose_style
                ], 
                outputs=[gen_video]
            )
        else:
            submit.click(
                fn=sad_talker.test, 
                inputs=[
                    source_image,
                    driven_audio,
                    preprocess_type,
                    is_still_mode,
                    enhancer,
                    batch_size,                            
                    size_of_image,
                    pose_style
                ], 
                outputs=[gen_video]
            )

    return sadtalker_interface

if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.launch(debug = True)
