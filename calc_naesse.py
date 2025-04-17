import cv2
import os
import numpy as np


def find_file(image_name):
    # Prompt the user to enter the file name
    # img_name_trocken = input("Enter the file name: ")

    # Get the current directory
    current_dir = os.getcwd()

    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Search for the file in every folder of the parent directory
    found = False
    for root, dirs, files in os.walk(parent_dir):
        if image_name in files:
            # If the file is found, print the full path to the file
            file_path = os.path.join(root, image_name)
            print(f"Found file: {file_path}")
            found = True
            break

    if not found:
        # If the file is not found, print an error message
        print(f"Error: File '{image_name}' not found in parent directory.")

    if found:
        img = cv2.imread(file_path, -1)
        img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    return img, img_gray


def get_bgr_values(image):
    # Initialize the minimum and maximum BGR values
    min_bgr = [255, 255, 255]
    max_bgr = [0, 0, 0]

    # Iterate over the pixels in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Get the BGR values of the pixel
            b, g, r = image[i, j]

            # Update the minimum and maximum BGR values if necessary
            min_bgr[0] = min(min_bgr[0], b)
            min_bgr[1] = min(min_bgr[1], g)
            min_bgr[2] = min(min_bgr[2], r)
            max_bgr[0] = max(max_bgr[0], b)
            max_bgr[1] = max(max_bgr[1], g)
            max_bgr[2] = max(max_bgr[2], r)

    # Print the minimum and maximum BGR values
    print(f"Minimum BGR values: {min_bgr}")
    print(f"Maximum BGR values: {max_bgr}")

    return min_bgr, max_bgr


def cut_circle(image, radius_add, show_img=True, save_img=False):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles in the gray image
    rows = img_gray.shape[0]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=100, param2=30,
                               minRadius=56, maxRadius=int(rows / 2))  # always: parameter 1 > parameter 2
    # param2: (number of points used to determine a circle?)

    # extract the radius and center of the largest detected circle (if any circle is found)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        idx_of_max_circle = circles[0, :, 2].argmax()
        i = circles[0, idx_of_max_circle]
        center = (i[0], i[1])
        # circle center
        # cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        # cv2.circle(img, center, radius+20, (255, 0, 255), 3)
    else:
        print('circles is none')

    height, width, depth = image.shape
    circle_img = np.zeros((height, width), np.uint8)

    # mask the image using the detected circle (radius is increased to more safely include the whole component)
    mask = cv2.circle(circle_img, center, radius + radius_add, (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(image, image, mask=circle_img)

    if show_img:
        cv2.imshow("masked_img", masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_circle_mask.tiff', masked_img)

    return masked_img


def grab_cut(image, radius_add, show_img=True, save_img=False):
    # Create a mask and a background model
    print("Initializing mask and background/foreground models...")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Select a rectangular region of interest (ROI) and apply grabcut
    print("Defining ROI and applying GrabCut algorithm...")
    rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterCount=15, mode=cv2.GC_INIT_WITH_RECT)

    # Get the foreground mask
    print("Extracting the foreground mask...")
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the original image by the foreground mask to get the segmented image
    print("Applying the foreground mask to the original image...")
    segmented_image = image * mask2[:, :, np.newaxis]

    # Convert the image to grayscale
    print("Converting the image to grayscale for circle detection...")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles in the grayscale image
    print("Detecting circles in the grayscale image...")
    rows = img_gray.shape[0]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=100, param2=30,
                               minRadius=56, maxRadius=int(rows / 2))

    # Extract the radius and center of the largest detected circle (if any circle is found)
    if circles is not None:
        print("Circles detected. Extracting the largest circle...")
        circles = np.uint16(np.around(circles))
        idx_of_max_circle = circles[0, :, 2].argmax()
        i = circles[0, idx_of_max_circle]
        center = (i[0], i[1])
        radius = i[2]
        print(f"Largest circle found at center: {center}, radius: {radius}")
    else:
        print("No circles detected.")
        return None

    # Create a blank image for masking
    print("Creating a blank image for masking...")
    height, width, depth = image.shape
    circle_img = np.zeros((height, width), np.uint8)

    # Mask the image using the detected circle
    print("Masking the image using the detected circle...")
    mask = cv2.circle(circle_img, center, radius + radius_add, (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(segmented_image, segmented_image, mask=circle_img)

    # Display the masked image if required
    if show_img:
        print("Displaying the masked image...")
        cv2.imshow("masked_img", masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the segmented image if required
    if save_img:
        print("Saving the masked image...")
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_BlackWhite_mask.tiff', masked_img)

    print("Returning the masked image...")
    return masked_img


def move_to_center(image, extra=2, show_img=True, save_img=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the Hough transform to detect circles in the image
    rows = image.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                                   param1=100, param2=30,
                                   minRadius=56, maxRadius=int(rows / 2))


    # extract the radius and center of the largest detected circle (if any circle is found)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        idx_of_max_circle = circles[0, :, 2].argmax()
        i = circles[0, idx_of_max_circle]
        x, y = (i[0], i[1])
        # circle center
        # cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        r = i[2]
        # cv2.circle(img, center, radius+20, (255, 0, 255), 3)
    else:
        print('circles is none')

    height, width, depth = image.shape
    circle_img = np.zeros((height, width), np.uint8)

    # Cut out the circle
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    circle_img = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the position to move the circle to
    x_pos = int(width / 2 - x)
    y_pos = int(height / 2 - y)

    # Create an empty image with the same size as the original image
    result = np.zeros_like(image)

    # Cut out a square the circle is in
    cut_square = circle_img[y-r-extra:y+r+extra, x-r-extra:x+r+extra]


    min_shape_value = min([image.shape[0], image.shape[1]])
    cut_square_res = cv2.resize(cut_square, (min_shape_value, min_shape_value), interpolation=cv2.INTER_CUBIC )

    # Move the circle to the middle of the image

    # Calculate Center-coordinates
    x_center= int(width / 2)
    y_center = int(height / 2)

    # Calculate Center
    x = x_center - (cut_square_res.shape[1]) // 2
    y = y_center - (cut_square_res.shape[0]) // 2

    # insert cut square into the middle
    result[y:y+cut_square_res.shape[0], x:x+cut_square_res.shape[1]] = cut_square_res

    # Show the image

    if show_img:
        cv2.imshow("Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_center.tiff', image)

    return result


def get_circles(image, show_img=True, save_img=False):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detects circles in the gray image
    rows = img_gray.shape[0]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=int(rows / 2))  # always: parameter 1 > parameter 2
    # param2: (number of points used to determine a circle?)

    # Make sure circles were detected
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the circles
        for (x, y, r) in circles:
            # Draw the circle on the image
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)

    if show_img:
        cv2.imshow("masked_img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_circles.tiff', image)

    return image


def mask_bgr_range(image, lower=[], higher=[], show_img=True, save_img=False):
    # mask out colors that fall within a specified range defined by "lower" and "higher" variables

    lower = np.array(lower)
    higher = np.array(higher)

    # add 3x3 blur to get a better result
    blur = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    final_trocken = cv2.inRange(blur.copy(), lower, higher)

    if show_img:
        cv2.imshow('final_trocken', final_trocken)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_BlackWhite_mask.tiff', final_trocken)

    return final_trocken


def morph_opening(image, kernelsize, show_img=True, save_img=False):
    # adds a morphology transformation called "opening" to remove noise

    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    if show_img:
        cv2.imshow('opening', opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_img:
        cv2.imwrite(f'..\\05_Waermebilder\\test\\{image}_opening.tiff', opening)

    return opening


def filter_trocken_nass(image_trocken, image_nass, show_img = True):
    # Read the images using cv2

    image1 = image_trocken
    image2 = image_nass

    # Get the size of image1
    height1, width1, _ = image1.shape

    # Get the size of image2
    height2, width2, _ = image2.shape

    # Check if the sizes are different
    if (height1, width1) != (height2, width2):
      # Resize image2 to the size of image1
      image1 = cv2.resize(image1, (width2, height2), interpolation=cv2.INTER_LINEAR)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Find the size of the images
    height, width = gray1.shape[:2]

    # Create a rotation matrix for each angle from 0 to 360 degrees
    rotation_matrices = [cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1) for angle in range(360)]

    # Rotate image1 using each rotation matrix and compare it to image2 using the normalized cross-correlation method
    best_angle = 0
    best_correlation = 0
    for angle, matrix in enumerate(rotation_matrices):
        rotated_image1 = cv2.warpAffine(gray1, matrix, (width, height))
        correlation = cv2.matchTemplate(rotated_image1, gray2, cv2.TM_CCORR_NORMED)[0][0]
        if correlation > best_correlation:
            best_angle = angle
            best_correlation = correlation

    # Rotate image1 by the best angle
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), best_angle, 1)
    rotated_image1 = cv2.warpAffine(image1, rotation_matrix, (width, height))

    # Create a mask for image1 by thresholding the rotated image
    ret, mask = cv2.threshold(cv2.cvtColor(rotated_image1, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Use the mask to filter image2
    filtered_image = cv2.bitwise_and(image2, image2, mask=mask_inv)

    # Subtract the filtered image from image2
    difference = cv2.subtract(image2, filtered_image)

    # Show the filtered image
    if show_img:
        cv2.imshow("img", difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return difference


def visualize(image_nass, masked_img, show_img=True, save_img=False):
    nass_filter = image_nass.copy()
    for i in range(image_nass.shape[0]):
        for j in range(image_nass.shape[1]):
            if masked_img[i][j] == 255:
                nass_filter[i][j] = [170, 216, 100]
            else:
                pass

    if show_img:
        cv2.imshow('nass_filter', nass_filter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #if save_img:
    #     cv2.imwrite(f'{img_name_nass_save}_vis.tiff', nass_filter)

    return nass_filter