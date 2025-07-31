import os
import json
import SimpleITK
import numpy as np
import torch

from pathlib import Path
from glob import glob

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

'''
docker run -it --rm --platform=linux/amd64 --network none --gpus all -v "/home/o644l/projects/panther/_test/input:/input" -v "/home/o644l/projects/panther/_test/output:/output" --entrypoint "" panther-dkfz-task2 /bin/bash
docker run -it --rm --platform=linux/amd64 --network none --gpus all -v "/home/o644l/projects/panther/_test/input:/input" -v "/home/o644l/projects/panther/_test/output:/output" panther-dkfz-task2
'''


print('All required modules are loaded!!!')

# PANTHER PATHS
# MRI_INPUT_PATH = Path("/input/images/ct/")
# CLICKS_INPUT_PATH = Path("/input/lesion-clicks.json")
# OUTPUT_PATH = Path("/output/images/tumor-lesion-segmentation/")

# LOCAL TEST
MRI_INPUT_PATH = Path("/input/images/abdominal-t2-mri/")
OUTPUT_PATH = Path("/output/images/pancreatic-tumor-segmentation/")

RESOURCE_PATH = Path("_model/task2")

def run():
    # Read the input
    input_array, spacing, direction, origin, uuid = load_image_file_as_array(
        location=MRI_INPUT_PATH,
    )
    
    input_array = input_array[None, ...] # (D, H, W) to (1, D, H, W) for nnUNet preproc
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    ############# Lines You can change ###########
    # Set the environment variable to handle memory fragmentation
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['nnUNet_compile'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    

    spacing_for_nnunet=list(spacing)[::-1]
    props = {
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
        # are returned x,y,z but spacing is returned z,y,x. Duh.
        'spacing': spacing_for_nnunet
    }
    print("Spacing is: ", spacing_for_nnunet)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True, # Changed last
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    predictor.initialize_from_trained_model_folder(RESOURCE_PATH, 
                                                    use_folds="all", 
                                                    checkpoint_name='checkpoint_final.pth')
    # Append None to the normalization schemes for the predictor's configuration manager
    # predictor.allowed_mirroring_axes = (1,2)

    import time
    start = time.time()
    ret = predictor.predict_single_npy_array(input_array, props, None, None, False)
    print("Time taken for prediction: ", time.time() - start)

    """
    import napari
    viewer = napari.Viewer()
    viewer.add_image(input_array[0], name='input')
    #viewer.add_image(np.flip(input_array, axis=3)[0], name='input_flipped')
    viewer.add_image(ret, name='output')
    #viewer.add_image(np.flip(ret, axis=2), name='output_flipped')
    napari.run()
    """


    ########## Don't Change Anything below this 
    # For some reason if you want to change the lines, make sure the output segmentation has the same properties (spacing, dimension, origin, etc) as the 
    # input volume
    ret[ret == 2] = 0
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH,
        array=ret,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
        uuid=uuid,
    )
    print('Saved!!!')
    return 0


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r") as json_file:
        return json.load(json_file)


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    print("Input files: ", input_files)
    # input_files = glob(str(location / "*.nii.gz")) ############## REMOVE THIS LINE IF YOU WANT TO USE MHA FILES ONLY
    uuid = os.path.splitext(os.path.basename(input_files[0]))[0]
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin, uuid



def write_array_as_image_file(*, location, array, spacing, origin, direction, uuid):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / Path(uuid + suffix),
        #useCompression=True,
    )


def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(torch.__version__)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())