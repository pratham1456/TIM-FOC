import cv2

def convert_to_grayscale(input_path, output_path):
    print(f"Input Path: {input_path}")
    print(f"Input Path: {output_path}")
    image = cv2.imread(input_path,0)
    if image is None:
        return "Failed to read image"

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(output_path, gray)
    # cv2.imwrite(output_path, image)
    message="Image converted to Black and White"
    return image, message

# convert_to_grayscale("captured.jpg", "detected.jpg")