import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def show_result(frame_0, generated_frames, n=10):

    plt.figure(figsize=(20, 4))
    for i in range(n):
        if i == 0 :
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(frame_0[i].reshape(128, 128, 4), cmap="gray")
            plt.axis("off")
        else:
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(generated_frames[i].reshape(128, 128, 4), cmap="gray")
            plt.axis("off")
    plt.show()


# def animate_frames(predicted_frames):
#     """
#     Create and display an animation from predicted frames in Jupyter Notebook.

#     Parameters:
#     predicted_frames (numpy array): Array of predicted frames, expected shape (num_frames, 128, 128, 3).
#     """
#     num_frames = predicted_frames.shape[0]

#     # Create a figure
#     fig, ax = plt.subplots()

#     # Display the first frame initially
#     img = ax.imshow(predicted_frames[0], animated=True)
#     ax.axis('off')  # Hide the axis for visual appeal

#     # Update function for the animation
#     def update(frame_num):
#         img.set_array(predicted_frames[frame_num])  # Update the image data with the new frame
#         return [img]

#     # Create the animation: FuncAnimation creates a new image every interval
#     ani = FuncAnimation(
#         fig, update, frames=num_frames, interval=200, blit=True  # interval sets the frame display time (in ms)
#     )

#     # Display the animation in Jupyter
#     plt.close(fig)  # Close the figure to prevent a static image from being displayed
#     return HTML(ani.to_jshtml())  # Display the animation in HTML format

# # saving gif file

# def animate_frames_s(predicted_frames, save_path=None, fps=5):
#     """
#     Create, display, and optionally save an animation from predicted frames in Jupyter Notebook.

#     Parameters:
#     predicted_frames (numpy array): Array of predicted frames, expected shape (num_frames, 128, 128, 3).
#     save_path (str, optional): Path to save the animation (supports formats like .mp4 or .gif).
#     fps (int, optional): Frames per second for the animation. Default is 5 fps.
#     """
#     num_frames = predicted_frames.shape[0]

#     # Create a figure
#     fig, ax = plt.subplots()

#     # Display the first frame initially
#     img = ax.imshow(predicted_frames[0], animated=True)
#     ax.axis('off')  # Hide the axis for visual appeal

#     # Update function for the animation
#     def update(frame_num):
#         img.set_array(predicted_frames[frame_num])  # Update the image data with the new frame
#         return [img]

#     # Create the animation: FuncAnimation creates a new image every interval
#     ani = FuncAnimation(
#         fig, update, frames=num_frames, interval=1000//fps, blit=True  # interval in ms
#     )

#     if save_path:
#         # Save the animation
#         if save_path.endswith(".mp4"):
#             ani.save(save_path, writer="ffmpeg", fps=fps)
#         elif save_path.endswith(".gif"):
#             ani.save(save_path, writer="pillow", fps=fps)
#         print(f"Animation saved to {save_path}")

#     # Display the animation in Jupyter
#     plt.close(fig)  # Close the figure to prevent a static image from being displayed
#     return HTML(ani.to_jshtml())

# def concat_images(idx, num_frames):
#   org = np.expand_dims(x_test[idx], axis=0) / 255
#   combined_images = np.concatenate((org, decoded_imgs[idx*10:idx*10+10]), axis=0)

#   return combined_images

# animate_frames(concat_images(0, 10))