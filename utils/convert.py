# Convert between NumPy array and Base64 string

import base64
import numpy as np
from io import BytesIO
from PIL import Image


def base64_to_numpy(base64_string):
    # Decode the Base64 string
    decoded_data = base64.b64decode(base64_string.split(',')[-1])

    # Convert the binary data to a NumPy array
    image_np = np.array(Image.open(BytesIO(decoded_data)))

    return image_np
