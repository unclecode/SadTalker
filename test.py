import requests

url = "http://localhost:8000/generate"

# Prepare the form data
form_data = {
    "text_input": "I'm a former founder who's reviewed over 30,000 pitches as a General Partner at Hustle Fund and as a former Partner at 500 Startups. This is a list of questions I regularly ask early-stage founders in order to decide whether or not to invest.",
    "voice_id": "oVYLyfQLNQCZVeuCq7oQ",
    "batch_size": "64",
    "pose_style": "2",
    "expression_scale": "1"
}

# Prepare the image file
image_file = "../SadTalker/examples/source_image/art_0.png"
files = {"source_image": open(image_file, "rb")}

# Send the POST request
response = requests.post(url, data=form_data, files=files)

# Save the output video
output_file = "output.mp4"
with open(output_file, "wb") as file:
    file.write(response.content)

print(f"Output video saved as {output_file}")