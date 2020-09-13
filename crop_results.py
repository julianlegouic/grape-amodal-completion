import os

from PIL import Image

workdir = os.path.join(os.getcwd(), 'results')
for subdir, dirs, files in os.walk(workdir):
    for i, filename in enumerate(files):
        # Check if the current file is an image (allow to avoid hidden files)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Create an Image object from an Image
            img = Image.open(os.path.join(subdir, filename))
            # Print progression
            print('Image info:',
                  '\nLocation:', os.path.join(subdir, filename),
                  '\nShape (w, h):', img.size)
            croppedimg = img.crop((26, 283, 26+1548, 283+1032))
            # Save the cropped image
            croppedimg.save(os.path.join(subdir, filename))

        # Free memory
        img.close()
        croppedimg.close()
