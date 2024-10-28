import base64

# Function to encode image to base64 string
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Encode the image
image_path = r"C:\Users\deeks\Downloads\FLOWER DETECTION\1.jpg" # Replace with the path to your image
base64_image = encode_image_to_base64(image_path)

# Print the Base64 string (or you can write it to a file)
print(base64_image)
