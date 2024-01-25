import torch
from PIL import Image
import numpy as np

def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def resize_image_max_size(image, max_size=768, multiple_of=8):
    width, height = image.size
    if width > height:
        max_width = max_size
        max_height = int((height / width) * max_size)
    else:
        max_width = int((width / height) * max_size)
        max_height = max_size

    max_width, max_height = resize_multiple_of((max_width, max_height), multiple_of)
    resized_image = image.resize((max_width, max_height))
    return resized_image

def resize_multiple_of(size, multiple_of=8):
    width, height = size
    new_width = (width // multiple_of) * multiple_of
    new_height = (height // multiple_of) * multiple_of
    return (new_width, new_height)

class ImageResizeAspectratio:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "max_size": ("INT", {
                    "default": 768, 
                    "min": 512, #Minimum value
                    "max": 2048, #Maximum value
                    "step": 1 #Slider's step
                }),
                "multiple_of": ("INT", {
                    "default": 8, 
                    "min": 1, #Minimum value
                    "max": 64, #Maximum value
                    "step": 1 #Slider's step
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "image_resize_aspectratio"

    #OUTPUT_NODE = False

    CATEGORY = "image"

    def image_resize_aspectratio(self, image, max_size, multiple_of):
        img = tensor_to_pil(image)
        img = resize_image_max_size(img, max_size, multiple_of)
        image = pil_to_tensor(img)
        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageResizeAspectratio": ImageResizeAspectratio
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeAspectratio": "ImageResizeAspectratio Node"
}
