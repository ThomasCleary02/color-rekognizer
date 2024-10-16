from collections import Counter

class RGBColorAnalyzer:
    def __init__(self):
        pass

    def analyze_image(self, img, num_colors=3):
        if img is None:
            raise ValueError("Error: Image not found or couldn't be read")

        # Resize image (optional, remove if you want to use original size)
        image = img.resize((700, 600))

        # Convert the image to RGB mode if it's not already
        image_rgb = image.convert('RGB')

        # Get all pixels
        pixels = list(image_rgb.getdata())

        # Count the occurrences of each color
        pixel_counts = Counter(pixels)

        # Calculate total number of pixels
        total_pixels = image.width * image.height

        # Get the most common colors and their counts
        most_common = pixel_counts.most_common(num_colors)

        # Convert to percentages and format the output
        color_percentages = {}
        for color, count in most_common:
            percentage = (count / total_pixels) * 100
            color_percentages[color] = round(percentage, 2)

        return color_percentages

    @staticmethod
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    @staticmethod
    def find_complement(rgb):
        return tuple(255 - value for value in rgb)