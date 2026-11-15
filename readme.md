# Running the motion video to turn image to video with ken burns effect
- *python motion_video.py image_path "slowly zoom in, fade in from black" <int_vid_duration> <x_focal_point> <y_focal_point>*
- *python src/motion_video.py images/image1.png "slowly zoom in, fade in from black" 24 700 378*


# image generator
- *python image_generator* 
- goes through all the scenes in inputs/storyboard.txt and makes an image using the prompt for it


# audio generator
- *python make_voice.py*
- goes through all the scenes in the storyboard and pulls out the *Script:* key for each and converts to audio