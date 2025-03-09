"""Functions to help create a custom dataset.

Run `help(CreateDataset)` to view the documentation of all functions.

Run this file directly to copy all the names of the fonts available in this
system to your clipboard. You can then save it to a file, say, 'fontnames.txt'.

The `FontDatasetCustom` class can be used.
"""

from base64 import b64encode
from io import BytesIO
from math import ceil
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------
#region Helper functions
# def get_font_size_for_height(height: int) -> float:
#     """Returns the font size to be set to obtain """

#     # https://www.desmos.com/calculator/ufxfft66fd
#     return 1.37309860932 * height -0.0482477220174

def load_fontnames(path: str) -> list[str]:
    """Load the font names from the given text file.

    Lines starting with a `#` are ignored."""

    with open(path, "r") as f:
        return [ls for line in f.readlines() if not (ls := line.strip()).startswith("#") and ls]

def font_from_fontname(fontname: str, size: float) -> ImageFont.FreeTypeFont:
    """Creates a `ImageFont.FreeTypeFont` object from the given font name."""

    filepath = font_manager.findfont(font_manager.FontProperties(family=fontname))
    return ImageFont.FreeTypeFont(font=filepath, size=size)

# imp
def text_render(font: str | ImageFont.FreeTypeFont, text: str, size: float, color=(255,255,255,255), bg=(0,0,0,0), transparency: bool=True) -> Image.Image:
    """Creates an image out of the given text.

    `font` can either be the name of the font, or a `FreeTypeFont` object."""

    if isinstance(font, str): font = font_from_fontname(font, size)
    x, y, r, b = font.getbbox(text) # x, y, right, bottom
    w, h = r-x, b-y

    img = Image.new("RGBA" if transparency else "RGB", (ceil(w), ceil(h)), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((-x, -y), text, fill=color, font=font)

    return img

## Below functions are for HTML display purposes
def img_to_data_url(img: Image.Image) -> str:
    """Creates a Data URL from the given PIL image object."""

    b = BytesIO()
    img.save(b, format="png")
    b64 = b64encode(b.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def img_to_html(img: Image.Image) -> str:
    """Creates a HTML `img` tag for the given PIL image object."""

    return f"<img src=\"{img_to_data_url(img)}\"/>"

def preview_html_text(fonts: list[str], text: str | None = None):
    """Generates a HTML table with the given font names with some sample text.

    The text is rendered as HTML text with the `font-family` set via CSS, so
    viewing the same code on different systems may give different outputs.

    If no sample text is given, then the name of the font is rendered."""

    return (
        "<style>td {text-align: left;} th {text-align: center; font-weight: bold;}</style>"
        + "<table><tr><th>Font name</th><th>Sample text</th></tr>"
        + "\n".join((
            f"<tr><td>{font}</td><td style=\"font-family: '{font}'\">{text or font}</td></tr>"
            for font in fonts
        ))
        + "</table>"
    )

def preview_html_image(fonts: list[str], text: str | None = None, size: float = 15, color=(255,255,255,255), bg=(0,0,0,0), transparency: bool=True):
    """Generates a HTML table with the given font names with some sample text.

    The text is rendered as images, so once the output is generated and, say,
    saved to a file or displayed in a Jupyter notebook, the result will appear
    the same in all systems."""

    lines = [
        "<style>td {text-align: left;} th {text-align: center; font-weight: bold;}</style>",
        "<table><tr><th>Font name</th><th>Sample text</th></tr>",
    ]
    for font in fonts:
        try: img = img_to_html(text_render(font, text or font, size, color, bg, transparency))
        except: img = "Error occurred"
        lines.append(f"<tr><td>{font}</td>"f"<td>{img}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)


## Below functions are used for image preparation for dataset/training/model
def get_top_bottom(arr, thresh: float, color: float):
    """Used by getbbox2. Gets the top and bottom indices after removing excess
    whitespace."""

    top, bottom = 0, arr.shape[0]

    for y, row in enumerate(arr):
        if abs(row - color).max() <= thresh: top = y
        else: break

    for y in range(arr.shape[0])[::-1]:
        row = arr[y]
        if abs(row - color).max() <= thresh: bottom = y
        else: break

    return top, bottom

def getbbox2(img: Image.Image, thresh: float=0.5, color=1.0) -> tuple[int, int, int, int]:
    """`getbbox` but then with any color, and with a threshold setting, i.e.
    lower threshold means all the pixels in a row/column should be closer to
    `color`.

    `color`: 0.0 = black, 1.0 = white."""

    pixels = (np.array(img) / 255.0) ** 2.2
    pixels_rotated = pixels.transpose()

    top, bottom = get_top_bottom(pixels, thresh, color)
    left, right = get_top_bottom(pixels_rotated, thresh, color)

    return (left, top, right, bottom)

## v v imp
def prepare_image(img: Image.Image, size: int, pad: int=2, color: int=255, thresh: float=0.5) -> Image.Image:
    """Does three things:
    1. Removes extra whitespace from all sides
    2. Resizes to square
    3. Adds a `pad`-px padding on all sides to reach the desired `size`

    `color` is a value from 0 to 255. Check `getbbox2` docstring for `thresh`
    """
    bbox = getbbox2(img, thresh, (color / 255.0) ** 2.2) # 1.0 = white
    return transforms.Pad(pad, (color,))(img.crop(bbox).resize((size-2*pad, size-2*pad)))

#endregion
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#region FontDatasetCustom class
class FontDatasetCustom(Dataset):
    """Contains the necessary logic and functions to generate a font dataset.

    Indexing the dataset will return a tuple:
    - image - a byte-array tensor of shape (1, img_dimension, img_dimension)
    - char_code - an integer denoting the Unicode codepoint
    - label - the font name
    """

    # The below three consist of every single character, where each index in
    # each of the following fields correspond to one character entry.
    data: list[torch.Tensor] # List of images (where each image is a tensor of shape (1, `img_dimension`, `img_dimension`))
    codes: list[int] # List of character codes (where each code is an int)
    labels: list[int] # List of labels (where each label is an int)

    # The transform sequence to be applied to an image:
    transform_image: transforms.Transform | None

    # The mapping used to represent a font name as a unique integer, "Font ID number" if you will.
    label_mapping: dict[str, int] # Font Name -> Font ID
    label_reverse_mapping: list[str] # Font ID -> Font name (basically just index the list normally)

    def __init__(self, characters: list[str], fontnames: list[str], fontsize: float | list[float], img_dimension: int=64, transform_image: transforms.Transform | None=None, color: int=255, bg: int=0):
        """
        `characters`: A list of characters.
        `fontnames`: A list of all font names.
        `fontsize`: The size of the text to render in. If you want to give multiple font sizes, give them as a list. An image will be created for each font size, and resized to `img_dimension`.
        `img_dimension`: The dimensions of the image that the model will receive.
        `transform_image`: The transform sequence used when an image is fetched.
        `color`: The text color.
        `bg`: The background color.
        """
        self.data = []
        labels_strs: list[str] = []
        self.codes = []
        self.transform_image = transform_image

        for fontname in fontnames:

            fontsizes = [fontsize] if isinstance(fontsize, (float, int)) else fontsize
            for size in fontsizes:
                try:
                    font = font_from_fontname(fontname, size)
                except:
                    print(f"WARNING: Font {fontname} not found. Skipping.")
                    break

                for char in characters:
                    # 1. Image of the current character
                    image = text_render(font, char, size, color=(color,color,color,255), bg=(bg,bg,bg,255), transparency=False)
                    image = image.convert("L")
                    image = prepare_image(image, img_dimension)
                    image_bytes = torch.tensor(np.array(image))

                    # 2. Character code (Unicode codepoint) of the character
                    char_code = ord(char) # same as char_obj["m_label"]

                    # 3. Font name of the character (i.e. "output")
                    label = fontname # Use font name as the label

                    self.data.append(image_bytes)
                    self.codes.append(char_code)
                    labels_strs.append(label)

                del font

        # Basically denote every font name with a number, useful for classification
        self.label_reverse_mapping = sorted(fontnames)
        self.label_mapping = {label: i for i, label in enumerate(self.label_reverse_mapping)}
        self.labels = [self.label_mapping[label] for label in labels_strs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        image = self.data[i]
        char_code = self.codes[i]
        label = self.labels[i]

        # Apply transformations every time a character is fetched
        if self.transform_image:
            image = self.transform_image(image)
        else:
            image = image.unsqueeze(0)

        return (image, char_code, label) # image is a byte-array tensor
#endregion
#-------------------------------------------------------------------------------

# Running this python script directly will print all the fonts installed in this
# system, and optionally copy it to the clipboard.
if __name__ == "__main__":
    print("Use following commands to get list of fonts:")
    print("   from matplotlib import font_manager")
    print("   font_manager.get_font_names()\n")

    from matplotlib import font_manager
    from pyperclip import copy

    fontnames_all = "\n".join(sorted(font_manager.get_font_names()))
    print(fontnames_all)
    if input("\nCopy to clipboard? [Y/n] ").strip().lower()[:1] != "n":
        copy(fontnames_all)
