"""
Python code for Questions 5 and 6:
Question5
1) Resolution Reduction
2) Gray-Level Reduction
3) Rotate an Image by a Specified Angle
Question6
4) Find the Set of Connected Pixels (Threshold-Based Adjacency)

"""

import math
import numpy as np
from PIL import Image
from collections import deque

def reduce_resolution(input_path, output_path, factor):
    """
    Down-sample (reduce resolution) by taking every 'factor'-th row and column.
    
    :param input_path:  Path to the input image (e.g. "Lena.tif")
    :param output_path: Path to save the reduced-resolution image (e.g. "Lena_reduced3.tif")
    :param factor:      Reduction factor (int). E.g. 3 or 5.
    """
    # Load image
    img = Image.open(input_path).convert("L")  # ensure grayscale
    arr = np.array(img)
    
    # Slice out every factor-th row/column
    reduced_arr = arr[::factor, ::factor]
    
    # Convert back to a PIL image
    out_img = Image.fromarray(reduced_arr)
    out_img.save(output_path)
    print(f"Saved down-sampled image to {output_path}")


def reduce_gray_levels(input_path, output_path, factor):
    """
    Reduce the number of gray levels by integer 'factor'.
    E.g. factor=2 => new pixel = floor(pixel/2)*2
    
    :param input_path:  Path to the input image
    :param output_path: Path to save the gray-level-reduced image
    :param factor:      Factor by which to reduce the gray levels (int)
    """
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.uint16)  # use 16-bit to avoid issues with multiplication
    
    # Quantize
    reduced_arr = (arr // factor) * factor
    
    # Back to 8-bit
    reduced_arr = np.clip(reduced_arr, 0, 255).astype(np.uint8)
    
    out_img = Image.fromarray(reduced_arr)
    out_img.save(output_path)
    print(f"Saved gray-level-reduced image to {output_path}")

def rotate_image(input_path, output_path, angle_degrees):
    """
    :param input_path:    Path to input image (e.g. "Lena.png")
    :param output_path:   Path to save the rotated image
    :param angle_degrees: Rotation angle in degrees (float). E.g. 30 for CCW.
    """

    # 1) Load image as grayscale array
    img_in = Image.open(input_path).convert("L")
    arr_in = np.array(img_in)
    h_in, w_in = arr_in.shape

    # 2) Prepare output array (same size as input, for simplicity)
    arr_out = np.zeros_like(arr_in, dtype=np.uint8)

    # 3) Compute center of the input image
    cx = w_in / 2.0
    cy = h_in / 2.0

    # 4) Convert angle to radians
    theta = math.radians(angle_degrees)

    # 5) For each pixel (x_out, y_out) in output:
    for y_out in range(h_in):
        for x_out in range(w_in):
            # Shift so (0,0) is at the image center:
            x_shifted = x_out - cx
            y_shifted = y_out - cy

            # Apply the INVERSE rotation to find where in the input it came from.
            # If the forward transform is:
            #   x =  v cosθ - w sinθ
            #   y =  v sinθ + w cosθ
            # Then the inverse rotation (solving for v,w) is:
            #   v =  x cosθ + y sinθ
            #   w = -x sinθ + y cosθ
            # We'll call (v_in, u_in) the input-plane coordinates.

            v_in = x_shifted * math.cos(theta) + y_shifted * math.sin(theta)
            u_in = -x_shifted * math.sin(theta) + y_shifted * math.cos(theta)

            # Shift back from center:
            col_in = v_in + cx
            row_in = u_in + cy

            # 6) Nearest-neighbor sampling:
            col_nn = int(round(col_in))
            row_nn = int(round(row_in))

            # 7) If it's inside the input bounds, copy the pixel
            if 0 <= row_nn < h_in and 0 <= col_nn < w_in:
                arr_out[y_out, x_out] = arr_in[row_nn, col_nn]
            else:
                arr_out[y_out, x_out] = 0  # or leave as zero background

    # 8) Save the result
    out_img = Image.fromarray(arr_out)
    out_img.save(output_path)
    print(f"Saved manually rotated image ({angle_degrees} deg) to {output_path}")


def connected_label(input_path, output_path, seed_row, seed_col, threshold):
    """
    Find all pixels that are connected to (seed_row, seed_col)
    via 4-adjacency, under the condition that neighboring pixels
    differ in intensity by no more than 'threshold'.
    
    The output is a label image (white=1, black=0).
    
    :param input_path:  Path to the input image (grayscale)
    :param output_path: Where to save the label image
    :param seed_row:    Row index of the seed pixel
    :param seed_col:    Column index of the seed pixel
    :param threshold:   Threshold for intensity difference (int)
    """
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.int32)
    rows, cols = arr.shape
    
    # Label array initialized to 0
    label = np.zeros((rows, cols), dtype=np.uint8)
    
    # BFS queue
    queue = deque()
    queue.append((seed_row, seed_col))
    
    while queue:
        r, c = queue.popleft()
        # If already labeled, skip
        if label[r, c] == 1:
            continue
        
        # Label it
        label[r, c] = 1
        
        # Check its 4 neighbors
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                if label[rr, cc] == 0:
                    # Compare intensities
                    if abs(arr[r, c] - arr[rr, cc]) <= threshold:
                        queue.append((rr, cc))
    
    # Convert label to 0/255 for easier viewing (white vs. black)
    out_arr = (label * 255).astype(np.uint8)
    
    out_img = Image.fromarray(out_arr)
    out_img.save(output_path)
    print(f"Saved connected-label image to {output_path}")


if __name__ == "__main__":
    """
    USAGE EXAMPLES:
    (Uncomment and adjust file paths/factors as needed)
    """
    
    # 1) Reduce resolution by factor 3, then 5
    #reduce_resolution("lena.tif", "Lena_res3.tif", factor=3)
    #reduce_resolution("lena.tif", "Lena_res5.tif", factor=5)
    
    # 2) Reduce gray levels by factor 2, 4, and 8
    #reduce_gray_levels("cman.tif", "Cameraman_gray2.tif", 2)
    #reduce_gray_levels("cman.tif", "Cameraman_gray4.tif", 4)
    #reduce_gray_levels("cman.tif", "Cameraman_gray8.tif", 8)
    
    # 3) Rotate Lena by 30 degrees CCW
    rotate_image("lena.tif", "Lena_rot30.tif", 30)
    
    # 4) Connected pixels for threshold T=4 from seed (67,45) on bird image
    #connected_label("birds.tif", "bird_connectedpixels_T4.tif", 67, 45, 4)
    
    pass
