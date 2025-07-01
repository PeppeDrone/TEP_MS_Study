import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# # Load the motion adapter
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# # load SD 1.5 based finetuned model
# model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
# pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
# scheduler = DDIMScheduler.from_pretrained(
#     model_id,
#     subfolder="scheduler",
#     clip_sample=False,
#     timestep_spacing="linspace",
#     beta_schedule="linear",
#     steps_offset=1,
# )
# pipe.scheduler = scheduler

# # enable memory savings
# pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

# output = pipe(
#     prompt=(
#         "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
#         "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
#         "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
#         "golden hour, coastal landscape, seaside scenery"
#     ),
#     negative_prompt="bad quality, worse quality",
#     num_frames=16,
#     guidance_scale=7.5,
#     num_inference_steps=25,
#     generator=torch.Generator("cpu").manual_seed(42),
# )
# frames = output.frames[0]
# export_to_gif(frames, "animation.gif")

# from diffusers import DiffusionPipeline
# from huggingface_hub import login
# import torch
# login('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')

# # Define the path to your local Stable Diffusion model
# model_path = "stable-3.5-large"  # Adjust if needed to reflect the full path

# # Load the pipeline using the local folder
# pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)

# # Move the model to CPU (or CUDA if you have a GPU)
# pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# # Generate an image from a text prompt
# prompt = "A newborn on a black background with legs down and rigth arm up"
# image = pipe(prompt).images[0]

# # Save the generated image
# image.save("generated_image.png")

# # Display the image
# image.show()
