import cv2 # to read and process video frames
from PIL import Image
import numpy as np
from scipy.cluster.vq import kmeans, vq
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# # Reference c/o GPT3.5
# # Define the colors you want to display
# colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

# # Convert the colors to RGB values
# rgb_colors = [mcolors.to_rgba(color) for color in colors]
# print(rgb_colors)
# # Create a 2D array of zeros with dimensions 1 x n, where n is the number of colors
# data = np.zeros((1, len(colors), 4))

# # Set the row to be the colors
# data[0] = rgb_colors

# # Display the colors
# plt.imshow(data, interpolation='nearest', aspect='auto')
# plt.axis('off')
# plt.show()

#%%

# Read the video file
# mp4, mkv extensions work
# .MOV not working

def process_vid(video_input):
	if video_input is not None:
		# video = cv2.VideoCapture('Green Screen video copy.mp4')
		print('video-input-data', video_input)
		video = cv2.VideoCapture(video_input)

		# Get the number of frames in the video
		num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		# Define the downscaled size
		size = (240, 135)

		def scan_frames(video_file, num_frames, downscale_size, frame_of_interest=int):
			# Initialize an empty list to store the dominant colors
			dominant_colors = []

			# Initialize variable frame_selected to be updated during the for loop
			frame_selected = ""

			for i in tqdm(range(num_frames)):

				# Read the frame
				success, frame = video_file.read()
				
				if not success:
						break

				# Pick the 1000th frame to be made as graph
				if i == frame_of_interest:
						frame_selected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						
				# Downscale the frame
				frame = cv2.resize(frame, downscale_size)
				
				# Convert the frame to a Pillow Image
				pil_image = Image.fromarray(frame[...,::-1])
				
				# Get the dominant color in the frame
				pixels = np.float32(pil_image).reshape(-1, 3)
				n_colors = 1
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
				flags = cv2.KMEANS_RANDOM_CENTERS
				_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
				_, counts = np.unique(labels, return_counts=True)
				dominant = palette[np.argmax(counts)]
				
				# Add the dominant color to the list
				dominant_colors.append(dominant)
			
			return frame_selected, dominant_colors

		# Call scan_frames function to start scanning
		frame_of_interest, dominant_colors = scan_frames(video, num_frames, size, 1000)

		# Convert the list to a NumPy array
		dominant_colors = np.array(dominant_colors)

		# Calculate the dominant color for the entire video
		codebook, _ = kmeans(dominant_colors, 5)

		def get_rgb(subarr):
				return mcolors.to_rgb(subarr/255)

		codebook_standardized = np.apply_along_axis(get_rgb, 1, codebook)
		rgb_colors = [mcolors.to_rgba(color) for color in codebook_standardized]

		data = np.zeros((1, len(codebook), 4))
		data[0] = rgb_colors

		# plot the color swatch and the frame as subplot objects
		fig, (ax1, ax2) = plt.subplots(2, 1)
		gs = gridspec.GridSpec(2,1, height_ratios=[2, 0.5])

		# Image frame from the for loop as graph
		ax1.imshow(frame_of_interest, interpolation='nearest', aspect='auto')
		fig.set_figheight(9)
		ax1.axis('off')

		# Color Swatch Graph
		ax2.imshow(data, interpolation='nearest', aspect='auto', extent=[ax1.get_xlim()[0], ax1.get_xlim()[1], ax2.get_ylim()[1], ax2.get_ylim()[0]])
		ax2.axis('off')

		# Reduce whitespace between
		fig.subplots_adjust(hspace=0)
		plt.tight_layout(h_pad=0, w_pad=0)
		# plt.show()

		# Release heap memory to ensure proper memory management
		video.release()
		cv2.destroyAllWindows()

		# Output on console in numpy array object and mean distance of all values from centroids
		# print(codebook, _, type(plt))

		return fig
	
