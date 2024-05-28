import PIL.Image

import sys


def merge_images(raw_img_file, left_img_file, right_img_file, output_file):
    raw_img = PIL.Image.open(raw_img_file)
    left_img = PIL.Image.open(left_img_file)
    right_img = PIL.Image.open(right_img_file)

    output_img = raw_img.copy()

    if output_img.mode == "I;16":
        output_img = output_img.point(lambda i: i * (1.0 / 256)).convert("RGB")
    print(output_img.size)
    print(left_img.size)
    print(right_img.size)
    output_img.paste(left_img, (53, 0))
    output_img.paste(right_img, (753, 0))

    output_img.save(output_file)


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python3 merge_image.py <path/to/original/image> <path/to/left/segment/plot> <path/to/right/segment/plot> <path/to/output_file>"
        )
        exit(1)

    raw_img_file = sys.argv[1]
    left_img_file = sys.argv[2]
    right_img_file = sys.argv[3]
    output_file = sys.argv[4]

    merge_images(raw_img_file, left_img_file, right_img_file, output_file)
